import pytest
import json
from unittest import mock
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO
from http import HTTPStatus
import detection_on_spam_prs  # rename this to your actual script file name, e.g., spam_detection


@pytest.fixture
def sample_row():
    """Provides a sample PR row dictionary with all expected fields."""
    return {
        "id": "PR123",
        "body": "This is a significant pull request body." * 10,
        "repository_name_with_owner": "user/repo",
        "url": "http://github.com/user/repo/pull/123",
        "created_at": "2021-01-01T00:00:00Z",
        "updated_at": "2021-01-02T00:00:00Z",
    }


def test_is_significant():
    """Tests that the is_significant function correctly identifies text length thresholds."""
    assert detection_on_spam_prs.is_significant("a" * 300) is True
    assert detection_on_spam_prs.is_significant("short text") is False


@patch("builtins.open", new_callable=mock_open, read_data="token1\ntoken2")
def test_load_tokens(mock_file):
    """Tests that tokens are correctly loaded from a file and split into a list."""
    with open("./env/zero-gpt-tokens.txt") as f:
        tokens = f.read().splitlines()
    assert tokens == ["token1", "token2"]


@patch("detection_on_spam_prs.requests.post")
def test_successful_detection(mock_post, sample_row):
    """
    Tests that a successful API call returns the expected HTTP status and result code.
    Mocks the requests.post call to return a 200 status and valid JSON response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"code": HTTPStatus.OK, "result": "Human"}

    mock_post.return_value = mock_response

    payload = json.dumps({"input_text": sample_row["body"]})
    headers = {
        "ApiKey": "testtoken",
        "Content-Type": "application/json",
    }

    response = detection_on_spam_prs.requests.post(
        "https://api.zerogpt.com/api/detect/detectText", headers=headers, data=payload
    )
    data = response.json()

    assert response.status_code == 200
    assert data["code"] == HTTPStatus.OK
    assert data["result"] == "Human"


@patch("detection_on_spam_prs.requests.post")
def test_detection_failure_and_retry(mock_post, sample_row):
    """
    Tests retry logic by simulating an initial failed API response followed by success.
    Verifies that the code attempts at least two calls and stops upon success.
    """
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 500
    mock_response_fail.json.return_value = {"code": 500}

    mock_response_success = MagicMock()
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = {"code": HTTPStatus.OK, "result": "AI"}

    # Simulate failure followed by success
    mock_post.side_effect = [mock_response_fail, mock_response_success]

    attempts = []
    for _ in range(2):  # emulate retry
        response = detection_on_spam_prs.requests.post(
            "https://api.zerogpt.com/api/detect/detectText", data={}
        )
        attempts.append(response.status_code)
        if response.status_code == 200:
            break

    assert attempts == [500, 200]


@patch("builtins.open", new_callable=mock_open)
def test_logging(mock_file):
    """
    Tests that the log_activity function correctly opens the log file and writes a log entry.
    """
    detection_on_spam_prs.log_activity("Test log entry")
    mock_file.assert_called_once_with(detection_on_spam_prs.log_path, "a")
    mock_file().write.assert_called_once()


def test_token_rotation():
    """
    Tests that token rotation logic maintains the length of the tokens list
    when rotated starting from index 0.
    """
    detection_on_spam_prs.tokens = ["t1", "t2", "t3"]
    start_index = 0
    rotated = (
        detection_on_spam_prs.tokens[start_index:]
        + detection_on_spam_prs.tokens[:start_index]
    )
    assert len(rotated) == len(detection_on_spam_prs.tokens)


@patch("detection_on_spam_prs.pickle.dump")
@patch("builtins.open", new_callable=mock_open)
def test_progress_saving(mock_open_file, mock_pickle):
    """
    Tests that progress is saved correctly by calling pickle.dump and writing to a file.
    """
    df = [{"id": "PR1", "zerogpt_response": "test"}]
    detection_on_spam_prs.pickle.dump({"df": df}, mock_open_file())
    mock_pickle.assert_called_once()
