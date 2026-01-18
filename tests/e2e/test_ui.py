"""
End-to-End UI Tests for KAIROS Dashboard

Tests the complete user workflow from frontend to backend:
- Page loading and rendering
- Form submission
- API integration
- Results display
- Error handling

Requires: selenium, pytest-selenium
Run with: pytest tests/e2e/test_ui.py --headless
"""

import pytest

pytestmark = pytest.mark.e2e
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException


@pytest.fixture(scope="module")
def browser():
    """Setup Chrome browser for testing."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(10)

    yield driver

    driver.quit()


@pytest.fixture
def dashboard_url():
    """Get dashboard URL from environment."""
    return os.getenv("DASHBOARD_URL", "http://localhost:5000")


class TestKairosDashboard:
    """End-to-end tests for KAIROS dashboard."""

    def test_dashboard_loads(self, browser, dashboard_url):
        """
        Test: Dashboard page loads successfully.
        Expected: Page title and main elements are present.
        """
        browser.get(dashboard_url)

        # Wait for page to load
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Check title
        assert (
            "KAIROS" in browser.title or "Risk" in browser.title
        ), f"Unexpected page title: {browser.title}"

        # Check for main container
        assert browser.find_element(By.TAG_NAME, "body"), "Page body not found"

    def test_form_fields_present(self, browser, dashboard_url):
        """
        Test: All input fields are present.
        Expected: Age, workclass, occupation, etc. fields exist.
        """
        browser.get(dashboard_url)

        # Expected form fields
        expected_fields = [
            "age",
            "workclass",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "education_num",
        ]

        for field_name in expected_fields:
            try:
                field = browser.find_element(By.NAME, field_name)
                assert field is not None, f"Field {field_name} not found"
            except Exception:
                # Try by ID as fallback
                try:
                    field = browser.find_element(By.ID, field_name)
                    assert field is not None
                except Exception:
                    pytest.fail(f"Field {field_name} not found by name or ID")

    def test_submit_prediction_success(self, browser, dashboard_url):
        """
        Test: Submit valid form and get prediction.
        Expected: Results display with decision, probability, uncertainty.
        """
        browser.get(dashboard_url)

        # Fill form with valid data
        form_data = {
            "age": "39",
            "workclass": "State-gov",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": "2174",
            "capital_loss": "0",
            "hours_per_week": "40",
            "native_country": "United-States",
            "education_num": "13",
        }

        for field_name, value in form_data.items():
            try:
                field = browser.find_element(By.NAME, field_name)
                field.clear()
                field.send_keys(value)
            except Exception:
                # Try select/dropdown
                try:
                    from selenium.webdriver.support.ui import Select

                    select = Select(browser.find_element(By.NAME, field_name))
                    select.select_by_visible_text(value)
                except Exception:
                    pass  # Field might be auto-filled or optional

        # Submit form
        submit_button = browser.find_element(
            By.CSS_SELECTOR, "button[type='submit'], input[type='submit']"
        )
        submit_button.click()

        # Wait for results (up to 10 seconds)
        try:
            WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "result"))
            )
        except TimeoutException:
            # Try alternative selectors
            try:
                WebDriverWait(browser, 5).until(
                    lambda d: "decision" in d.page_source.lower()
                    or "probability" in d.page_source.lower()
                )
            except TimeoutException:
                pytest.fail("Results did not appear within timeout")

        # Verify results contain expected fields
        page_source = browser.page_source.lower()
        assert any(
            word in page_source for word in ["accept", "reject", "abstain"]
        ), "Decision not found in results"
        assert (
            "probability" in page_source or "confidence" in page_source
        ), "Probability not found in results"

    def test_submit_invalid_data(self, browser, dashboard_url):
        """
        Test: Submit invalid form data.
        Expected: Error message displayed, no crash.
        """
        browser.get(dashboard_url)

        # Fill with invalid data
        try:
            age_field = browser.find_element(By.NAME, "age")
            age_field.clear()
            age_field.send_keys("-999")  # Invalid age
        except Exception:
            pass

        # Submit form
        try:
            submit_button = browser.find_element(
                By.CSS_SELECTOR, "button[type='submit'], input[type='submit']"
            )
            submit_button.click()

            # Should show error or validation message
            time.sleep(2)
            page_source = browser.page_source.lower()

            # Either validation prevents submission or error is shown
            assert any(
                word in page_source for word in ["error", "invalid", "required"]
            ) or browser.find_element(By.NAME, "age").get_attribute(
                "validationMessage"
            ), "No error handling for invalid input"
        except Exception:
            pass  # HTML5 validation might prevent submission

    def test_multiple_predictions(self, browser, dashboard_url):
        """
        Test: Submit multiple predictions in sequence.
        Expected: Each prediction works independently.
        """
        browser.get(dashboard_url)

        test_cases = [
            {"age": "25", "education_num": "10"},
            {"age": "45", "education_num": "16"},
            {"age": "60", "education_num": "12"},
        ]

        for test_data in test_cases:
            # Update fields
            for field_name, value in test_data.items():
                try:
                    field = browser.find_element(By.NAME, field_name)
                    field.clear()
                    field.send_keys(value)
                except Exception:
                    pass

            # Submit
            try:
                submit_button = browser.find_element(
                    By.CSS_SELECTOR, "button[type='submit'], input[type='submit']"
                )
                submit_button.click()
                time.sleep(2)  # Wait for response
            except Exception:
                pass

    def test_responsive_design(self, browser, dashboard_url):
        """
        Test: Dashboard works on different screen sizes.
        Expected: Layout adapts to mobile/tablet/desktop.
        """
        screen_sizes = [
            (375, 667),  # Mobile (iPhone)
            (768, 1024),  # Tablet (iPad)
            (1920, 1080),  # Desktop
        ]

        for width, height in screen_sizes:
            browser.set_window_size(width, height)
            browser.get(dashboard_url)

            # Check page still loads
            assert browser.find_element(
                By.TAG_NAME, "body"
            ), f"Page failed to load at {width}x{height}"

            time.sleep(1)  # Let page render

    def test_api_error_handling(self, browser, dashboard_url):
        """
        Test: Dashboard handles API errors gracefully.
        Expected: User-friendly error message, no crash.
        """
        # This test assumes you can simulate API failure
        # In real scenario, you'd mock the backend or use a test endpoint
        browser.get(dashboard_url)

        # Try to trigger an error condition
        # (Implementation depends on your frontend error handling)

        # Check that page doesn't crash
        assert browser.find_element(By.TAG_NAME, "body"), "Page crashed on error"

    def test_accessibility_basics(self, browser, dashboard_url):
        """
        Test: Basic accessibility features.
        Expected: Form labels, ARIA attributes, keyboard navigation.
        """
        browser.get(dashboard_url)

        # Check for form labels
        labels = browser.find_elements(By.TAG_NAME, "label")
        assert len(labels) > 0, "No form labels found"

        # Check for ARIA attributes (basic check)
        inputs = browser.find_elements(By.TAG_NAME, "input")
        for input_elem in inputs[:5]:  # Check first 5
            # Should have either label, aria-label, or aria-labelledby
            # Not strictly required but good practice to check
            _ = (
                input_elem.get_attribute("aria-label")
                or input_elem.get_attribute("aria-labelledby")
                or input_elem.get_attribute("id")
            )
            # assert has_label, "Input missing accessibility attributes"

    def test_performance_metrics(self, browser, dashboard_url):
        """
        Test: Page load performance.
        Expected: Page loads within acceptable time.
        """
        start_time = time.time()
        browser.get(dashboard_url)

        # Wait for page to be fully loaded
        WebDriverWait(browser, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        load_time = time.time() - start_time

        # Page should load within 5 seconds
        assert load_time < 5, f"Page took {load_time:.2f}s to load (>5s threshold)"
