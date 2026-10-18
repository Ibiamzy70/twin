from digital_twin.extraction import (
    extract_patterns,
    extract_profile_data,
)

# --------------------------------------------------
# extract_patterns() tests
# --------------------------------------------------


def test_extract_email():
    text = "Contact me at john.doe@email.com"

    result = extract_patterns(text, "email")

    assert result == ["john.doe@email.com"]


def test_extract_phone():
    text = "Call me at +1-555-123-4567"

    result = extract_patterns(text, "phone")

    assert result == ["+1-555-123-4567"]


def test_extract_date():
    text = "The project starts on 2026-03-01."

    result = extract_patterns(text, "date")

    assert result == ["2026-03-01"]


def test_extract_time():
    text = "I usually wake up at 6:30am."

    result = extract_patterns(text, "time")

    assert result == ["6:30am"]


def test_extract_url():
    text = "Visit https://example.com/profile"

    result = extract_patterns(text, "url")

    assert result == ["https://example.com/profile"]


def test_extract_hashtag():
    text = "#running #fitness #health"

    result = extract_patterns(text, "hashtag")

    assert result == ["#running", "#fitness", "#health"]


def test_extract_mention():
    text = "Follow @SarahRuns on Twitter."

    result = extract_patterns(text, "mention")

    assert result == ["@SarahRuns"]


def test_mention_does_not_extract_email():
    text = "Email me at sarah.runner@email.com"

    result = extract_patterns(text, "mention")

    assert result == []


def test_extract_number():
    text = "I ran 5km and completed 10.5 kilometers."

    result = extract_patterns(text, "number")

    assert result == ["5", "10.5"]


def test_extract_duration():
    text = "The run took 30min and the workout lasted 2 hours."

    result = extract_patterns(text, "duration")

    assert result == ["30min", "2 hours"]


def test_unknown_pattern_raises_error():
    text = "Some text"

    try:
        extract_patterns(text, "unknown")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Unknown pattern" in str(exc)


# --------------------------------------------------
# extract_profile_data() tests
# --------------------------------------------------


def test_extract_profile_data():
    text = """
    Hi! I'm Sarah (sarah.runner@email.com).
    Reach me at +1-555-234-5678.

    I run 5km daily at 6:30am and love
    #running #fitness #health.

    My goal is 10km in 50min by 2026-03-01.

    Follow me at @SarahRuns.
    """

    result = extract_profile_data(text)

    assert result["identifiers"]["emails"] == ["sarah.runner@email.com"]

    assert result["identifiers"]["phones"] == ["+1-555-234-5678"]

    assert result["identifiers"]["handles"] == ["@SarahRuns"]

    assert result["temporal"]["dates"] == ["2026-03-01"]

    assert result["temporal"]["times"] == ["6:30am"]

    assert result["temporal"]["durations"] == [
        "5km",
        "50min",
    ]

    assert result["preferences"]["interests"] == [
        "running",
        "fitness",
        "health",
    ]


def test_profile_data_returns_expected_structure():
    result = extract_profile_data(
        "Email: test@example.com. " "Meet me at 8:00am. #python"
    )

    assert "identifiers" in result
    assert "temporal" in result
    assert "preferences" in result

    assert "emails" in result["identifiers"]
    assert "phones" in result["identifiers"]
    assert "handles" in result["identifiers"]

    assert "dates" in result["temporal"]
    assert "times" in result["temporal"]
    assert "durations" in result["temporal"]

    assert "interests" in result["preferences"]


def test_empty_text_returns_empty_results():
    result = extract_profile_data("")

    assert result == {
        "identifiers": {
            "emails": [],
            "phones": [],
            "handles": [],
        },
        "temporal": {
            "dates": [],
            "times": [],
            "durations": [],
        },
        "preferences": {
            "interests": [],
        },
    }


def test_non_string_input_raises_type_error():
    try:
        extract_profile_data(None)
        assert False, "Expected TypeError"
    except TypeError as exc:
        assert "text must be a string" in str(exc)


def test_extract_patterns_non_string_input_raises_type_error():
    try:
        extract_patterns(None, "email")
        assert False, "Expected TypeError"
    except TypeError as exc:
        assert "text must be a string" in str(exc)
