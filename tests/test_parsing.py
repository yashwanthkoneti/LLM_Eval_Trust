"""The deterministic layer. These are the bugs that cost the most to find."""

from policy_audit.parsing.chunks import parse_chunks, parse_fields, split_chunks
from policy_audit.parsing.clarification import (answers_an_option,
                                                detect_askback,
                                                parse_clarification)
from policy_audit.parsing.ontology import build_ontology
from policy_audit.parsing.tables import parse_table, split_row
from policy_audit.domain.models import Turn

MARKERS = ("please clarify",)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def test_split_row_keeps_empty_leading_cell():
    """`.strip('|')` eats it and every column shifts left by one."""
    assert split_row("||Amount|Registration|") == ["", "Amount",
                                                   "Registration"]


def test_split_row_keeps_empty_trailing_cell():
    assert split_row("|Amount|Registration||") == ["Amount", "Registration",
                                                   ""]


def test_split_row_keeps_empty_middle_cell():
    assert split_row("|Band||Approver|") == ["Band", "", "Approver"]


def test_escaped_pipe_is_data():
    assert split_row(r"|Range|$25 \| $200|") == ["Range", "$25 | $200"]


def test_table_alignment_survives_blank_header_cell():
    table = parse_table(["||Amount|Registration required|",
                         "|:---|---:|:--:|",
                         "|Gifts|$25 and up to $200|No|",
                         "|Gifts|above $200|Yes|"])
    assert table is not None
    assert table.header == ("col1", "Amount", "Registration required")
    rendered = table.as_rows()
    # the value must stay attached to its OWN column
    assert "Amount=$25 and up to $200" in rendered[0]
    assert "Registration required=No" in rendered[0]
    assert "Registration required=Yes" in rendered[1]


def test_short_row_is_padded_not_slid_left():
    table = parse_table(["|A|B|C|", "|1|2|"])
    assert table is not None
    assert table.rows[0] == ("1", "2", "")


def test_separator_only_row_is_dropped_but_blank_row_is_not_a_separator():
    table = parse_table(["|A|B|", "|---|---|", "|  |  |", "|1|2|"])
    assert table is not None
    assert table.rows == (("1", "2"),)


def test_single_pipe_row_is_not_a_table():
    assert parse_table(["|just one row|"]) is None


# --------------------------------------------------------------------------
# chunks
# --------------------------------------------------------------------------
RAW = ("**__FILE_NAME__:**PAD_Std_v4.pdf**__DOCUMENT_TITLE__:**Personal "
       "Account Dealing Standard**__SECTION__:**4.2 Pre-clearance"
       "**__TEXT__:**Limits by region:\n"
       "|Region|Threshold|Approver|\n|---|---|---|\n"
       "|EMEA|USD 50,000|Regional CO|\n|APAC|USD 25,000|Global CO|\n"
       "Broad-based ETFs are exempt from the above.,"
       "**__FILE_NAME__:**GE_Pol_v2.pdf**__DOCUMENT_TITLE__:**Gifts and "
       "Entertainment Policy**__TEXT__:**Gifts above USD 100 need approval.")


def test_decorated_keys_are_undecorated():
    """`__FILE_NAME__` must become something a lookup can find.

    Case is kept so the rendered chunk reads naturally; lookups are
    case-insensitive. What must not survive is the underscores, which is
    what made every title resolve to "?".
    """
    fields = parse_fields("**__FILE_NAME__:**x.pdf")
    assert fields[0][0].lower() == "file name"


def test_field_lookup_is_case_and_shape_insensitive():
    chunk = parse_chunks("**__FILE_NAME__:**x.pdf")[0]
    assert chunk.field("file name") == "x.pdf"
    assert chunk.field("filename", "file name") == "x.pdf"


def test_chunks_split_on_repeated_key_not_on_commas():
    assert len(split_chunks(RAW)) == 2


def test_title_resolves_rather_than_becoming_a_question_mark():
    chunks = parse_chunks(RAW)
    assert chunks[0].doc_title == "Personal Account Dealing Standard"
    assert chunks[1].doc_title == "Gifts and Entertainment Policy"


def test_table_is_extracted_and_prose_order_preserved():
    chunk = parse_chunks(RAW)[0]
    assert len(chunk.tables) == 1
    body = chunk.text
    # the sentence that FOLLOWED the table must still follow it, or a
    # qualifier lands against the wrong rows
    assert body.index("row 1") < body.index("Broad-based ETFs are exempt")


def test_value_commas_do_not_break_parsing():
    raw = "**__ALPHA__:**one, two, three**__BETA__:**four"
    fields = {k.lower(): v for k, v in parse_fields(raw)}
    assert fields["alpha"] == "one, two, three"
    assert fields["beta"] == "four"


# --------------------------------------------------------------------------
# clarification and askback
# --------------------------------------------------------------------------
TEMPLATE = """To help you better, please clarify:

1. Which policy area are you asking about ?

option a: Personal Account Dealing
option b: Gifts and Entertainment
option c: Outside Business Interests

2. Which region applies to you ?

option a: EMEA
option b: APAC
option c: Americas

3. Are you an employee or a reviewer ?

option a: Employee
option b: Reviewer
option c: Other"""


def test_template_parses_three_questions_and_nine_options():
    c = parse_clarification(TEMPLATE, MARKERS)
    assert c.is_clarification
    assert c.n_questions == 3
    assert c.n_options == 9


def test_question_without_a_question_mark_is_still_a_question():
    c = parse_clarification("Please clarify:\n1. Which region\noption a: EMEA",
                            MARKERS)
    assert c.n_questions == 1
    assert c.n_options == 1


def test_option_with_empty_text_still_counts():
    c = parse_clarification("Please clarify:\n1. Which region ?\n"
                            "option a:\noption b:", MARKERS)
    assert c.n_options == 2


def test_no_marker_means_not_a_clarification():
    c = parse_clarification("Could you tell me which region you mean?",
                            MARKERS)
    assert not c.is_clarification


def test_askback_is_detected_from_the_previous_assistant_turn():
    history = (Turn("user", "What leave am I entitled to?"),
               Turn("assistant", TEMPLATE))
    is_askback, asked = detect_askback(history, MARKERS)
    assert is_askback
    assert "Which policy area" in asked


def test_askback_false_when_previous_turn_was_a_normal_answer():
    history = (Turn("user", "What leave am I entitled to?"),
               Turn("assistant", "You are entitled to 25 days."))
    assert detect_askback(history, MARKERS)[0] is False


def test_request_matching_an_offered_option_is_identified():
    history = (Turn("user", "What leave?"), Turn("assistant", TEMPLATE))
    assert answers_an_option("Gifts and Entertainment", history,
                             MARKERS) == "Gifts and Entertainment"


# --------------------------------------------------------------------------
# ontology
# --------------------------------------------------------------------------
def test_ontology_names_the_dimension_a_table_varies_by():
    ont = build_ontology(parse_chunks(RAW))
    assert "Personal Account Dealing Standard" in ont.documents
    assert "Region" in ont.table_columns
    # the screener must be told the policy varies by region, or an answer
    # that names no region looks fine
    assert any("region" in d.lower() for d in ont.conditional_dimensions)
    assert ont.alternative_count >= 2


def test_ontology_is_empty_without_retrieval():
    ont = build_ontology([])
    assert ont.documents == ()
    assert ont.conditional_dimensions == ()
    assert "nothing retrieved" in ont.render()


def test_unconditional_single_row_table_is_not_a_conditional_axis():
    raw = ("**__DOCUMENT_TITLE__:**Fees**__TEXT__:**"
           "|Fee|Amount|\n|---|---|\n|Annual|USD 100|")
    ont = build_ontology(parse_chunks(raw))
    assert ont.alternative_count == 0


# --------------------------------------------------------------------------
# table forms beyond the plain pipe case
# --------------------------------------------------------------------------
def test_blank_line_inside_a_table_does_not_end_it():
    """It used to split the header off the body, losing every binding."""
    from policy_audit.parsing.tables import split_segments
    tables = [p for k, p in split_segments(
        "|Band|Approver|\n|---|---|\n\n|up to 10k|Manager|") if k == "table"]
    assert tables
    assert "Approver=Manager" in tables[0].as_rows()[0]


def test_tab_separated_rows_parse():
    from policy_audit.parsing.tables import split_segments
    tables = [p for k, p in split_segments(
        "Band\tApprover\nup to 10k\tManager\nabove 10k\tDivision Head")
        if k == "table"]
    assert tables
    assert tables[0].header == ("Band", "Approver")
    assert "Approver=Division Head" in tables[0].as_rows()[1]


def test_unparseable_table_is_flagged_not_silently_dropped():
    """The judge must know a table resisted structuring."""
    html = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**"
                        "<table><tr><th>Band</th></tr>"
                        "<tr><td>10k</td></tr></table>")[0]
    spaced = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**"
                          "Band           Approver\n"
                          "up to 10k      Manager\n"
                          "above 10k      Division Head")[0]
    assert html.has_unstructured_table
    assert spaced.has_unstructured_table


def test_prose_and_parsed_tables_are_not_flagged():
    prose = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**The approver "
                         "is your line manager below the threshold.")[0]
    piped = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**"
                         "|Band|Approver|\n|---|---|\n|10k|Mgr|")[0]
    assert not prose.has_unstructured_table
    assert not piped.has_unstructured_table


def test_ontology_still_finds_the_axis_without_the_prose_scan():
    """The table says it; the prose scan was paying 1.5ms to agree."""
    ont = build_ontology(parse_chunks(
        "**__DOCUMENT_TITLE__:**Approval**__TEXT__:**"
        + "Narrative that mentions region and entity at length. " * 50
        + "\n|Band|Approver|\n|---|---|\n|up to 10k|Manager|\n"
          "|above 10k|Division Head|"))
    assert ont.conditional_dimensions == ("Band",)
    assert ont.alternative_count == 2
