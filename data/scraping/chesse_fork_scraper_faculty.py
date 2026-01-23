"""
Enhanced Cheesefork Course Information Scraper
Discovers and scrapes courses by prefix with recovery and progress tracking
WITH COMPREHENSIVE DEBUG LOGGING
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class CheeseforkScraper:
    def __init__(self, headless=True, debug=True):
        """Initialize the scraper with Chrome driver"""
        self.debug = debug
        self.log_debug("Initializing Chrome driver...")

        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # Add more options to ensure clean state
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--incognito")  # Start in incognito mode for clean state

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)
        self.base_url = "https://cheesefork.cf/"
        self.log_debug("Chrome driver initialized successfully")

    def log_debug(self, message):
        """Log debug messages if debug mode is enabled"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_debug("Closing Chrome driver...")
        try:
            self.driver.quit()
            self.log_debug("Chrome driver closed successfully")
        except Exception as e:
            self.log_debug(f"Error closing driver: {e}")

    def load_page(self):
        """Load the main cheesefork page"""
        self.log_debug(f"Loading page: {self.base_url}")
        self.driver.get(self.base_url)

        self.log_debug("Setting localStorage to skip tutorial...")
        # Set localStorage to skip tutorial on future visits
        try:
            self.driver.execute_script("""
                localStorage.setItem('dontShowTip', 'true');
                localStorage.setItem('dontShowHistogramsTip', Date.now().toString());
            """)
            self.log_debug("localStorage set successfully")
        except Exception as e:
            self.log_debug(f"Error setting localStorage: {e}")

        time.sleep(3)
        self.dismiss_intro_tutorial()
        self.log_debug("Page loaded and ready")

    def dismiss_intro_tutorial(self):
        """Dismiss the intro.js tutorial overlay if present"""
        self.log_debug("Attempting to dismiss tutorial overlay...")
        try:
            time.sleep(1)

            skip_selectors = [
                ".introjs-skipbutton",
                ".introjs-button.introjs-skipbutton",
                "a.introjs-skipbutton",
                ".introjs-donebutton",
                ".introjs-button.introjs-donebutton"
            ]

            for selector in skip_selectors:
                try:
                    skip_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if skip_button.is_displayed():
                        self.log_debug(f"Found skip button with selector: {selector}")
                        skip_button.click()
                        time.sleep(1)
                        self.log_debug("Tutorial dismissed successfully")
                        return True
                except NoSuchElementException:
                    continue

            # Try ESC key
            try:
                from selenium.webdriver.common.keys import Keys
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.send_keys(Keys.ESCAPE)
                time.sleep(0.5)
                self.log_debug("Sent ESC key to dismiss tutorial")
                return True
            except:
                pass

            # Try clicking overlay
            try:
                overlay = self.driver.find_element(By.CSS_SELECTOR, ".introjs-overlay")
                if overlay.is_displayed():
                    overlay.click()
                    time.sleep(0.5)
                    self.log_debug("Clicked overlay to dismiss tutorial")
                    return True
            except:
                pass

            self.log_debug("No tutorial overlay found (this is OK)")
            return False

        except Exception as e:
            self.log_debug(f"Error dismissing tutorial: {e}")
            return False

    def discover_courses_by_prefix(self, prefix):
        """Discover all courses that start with the given prefix"""
        self.log_debug(f"Discovering courses with prefix {prefix}...")

        try:
            # Find the course select dropdown
            self.log_debug("Looking for course input field...")
            course_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".course-select .selectize-input input"))
            )
            self.log_debug("Found course input field")

            # Clear any previous input
            course_input.clear()

            # Click to focus
            course_input.click()
            time.sleep(0.5)

            # Type prefix
            self.log_debug(f"Typing prefix: {prefix}")
            course_input.send_keys(prefix)
            time.sleep(2)  # Wait for dropdown results

            # Wait for dropdown options to appear
            self.log_debug("Waiting for dropdown to appear...")
            dropdown = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".selectize-dropdown-content"))
            )
            self.log_debug("Dropdown appeared")

            time.sleep(1)  # Wait for all options to render

            # Find all course options
            options = dropdown.find_elements(By.CSS_SELECTOR, ".option")
            self.log_debug(f"Found {len(options)} options in dropdown")

            discovered_courses = []

            for i, option in enumerate(options):
                try:
                    option_value = option.get_attribute("data-value")
                    option_text = option.text

                    # Skip the "partial" indicator
                    if option_value == "partial":
                        continue

                    # Extract course ID from the option value
                    if option_value:
                        if len(option_value) >= 8:
                            course_id = option_value[1:4] + option_value[5:8]

                            discovered_courses.append({
                                "course_id": course_id,
                                "name": option_text,
                                "data_value": option_value
                            })

                except Exception as e:
                    self.log_debug(f"Error processing option {i}: {e}")
                    continue

            self.log_debug(f"Discovered {len(discovered_courses)} courses with prefix {prefix}")

            # Clear the search to reset for next discovery
            course_input.clear()
            time.sleep(0.5)

            return discovered_courses

        except TimeoutException:
            self.log_debug(f"TIMEOUT while discovering courses for prefix {prefix}")
            return []
        except Exception as e:
            self.log_debug(f"ERROR discovering courses: {str(e)}")
            return []

    def search_and_add_course(self, course_id):
        """Search for course and add it to the schedule"""
        self.log_debug(f"Searching for course: {course_id}")
        try:
            self.log_debug("Finding course input field...")
            course_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".course-select .selectize-input input"))
            )
            self.log_debug("Found course input field")

            course_input.click()
            time.sleep(0.5)

            self.log_debug(f"Typing course ID: {course_id}")
            course_input.send_keys(str(course_id))
            time.sleep(2)

            self.log_debug("Waiting for dropdown...")
            dropdown = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".selectize-dropdown-content"))
            )
            self.log_debug("Dropdown appeared")

            time.sleep(1)

            options = dropdown.find_elements(By.CSS_SELECTOR, ".option")
            self.log_debug(f"Found {len(options)} options in dropdown")

            course_found = False

            s = "0" + str(course_id)
            result = s[:4] + "0" + s[4:]
            self.log_debug(f"Looking for transformed ID: {result}")

            for i, option in enumerate(options):
                try:
                    option_value = option.get_attribute("data-value")

                    if option_value == "partial":
                        continue

                    if option_value and result in option_value:
                        self.log_debug(f"Found matching option at index {i}: {option_value}")
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", option)
                        time.sleep(0.3)

                        try:
                            option.click()
                            self.log_debug("Clicked option successfully")
                        except:
                            self.driver.execute_script("arguments[0].click();", option)
                            self.log_debug("Clicked option using JavaScript")

                        course_found = True
                        time.sleep(1)
                        break
                except Exception as e:
                    self.log_debug(f"Error processing option {i}: {e}")
                    continue

            if course_found:
                self.log_debug(f"✓ Successfully added course {course_id}")
            else:
                self.log_debug(f"✗ Could not find course {course_id}")

            return course_found

        except TimeoutException:
            self.log_debug(f"✗ TIMEOUT searching for course {course_id}")
            return False
        except Exception as e:
            self.log_debug(f"✗ ERROR searching for course {course_id}: {e}")
            return False

    def transform_course_id(self, course_id):
        """Transform course ID to match website format"""
        s = "0" + str(course_id)
        result = s[:4] + "0" + s[4:]
        return result

    def click_info_button(self, course_id):
        """Click the 'i' button for the course"""
        self.log_debug(f"Clicking info button for course {course_id}")
        try:
            transformed_id = self.transform_course_id(course_id)
            self.log_debug(f"Looking for course button with ID: {transformed_id}")

            course_button = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, f'li.list-group-item[data-course-number="{transformed_id}"]')
                )
            )
            self.log_debug("Found course button")

            info_badge = course_button.find_element(By.CSS_SELECTOR, ".badge")
            self.log_debug("Found info badge")

            info_badge.click()
            self.log_debug("Clicked info badge")
            time.sleep(2)

            dialog = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".bootstrap-dialog"))
            )
            self.log_debug("✓ Info dialog opened successfully")
            return True

        except TimeoutException:
            self.log_debug(f"✗ TIMEOUT clicking info button for course {course_id}")
            return False
        except Exception as e:
            self.log_debug(f"✗ ERROR clicking info button: {e}")
            return False

    def extract_course_information(self):
        """Extract course information from the dialog"""
        self.log_debug("Extracting course information...")
        try:
            course_info_section = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".course-information"))
            )

            info_text = course_info_section.text

            course_data = {
                "raw_text": info_text
            }

            try:
                course_data["title"] = course_data["raw_text"].splitlines()[0]
            except:
                course_data["title"] = "Unknown"

            self.log_debug(f"✓ Extracted course information: {course_data['title']}")
            return course_data

        except Exception as e:
            self.log_debug(f"✗ ERROR extracting course information: {e}")
            return {}

    def extract_feedback(self):
        """Extract course feedback and reviews"""
        self.log_debug("Extracting feedback...")
        try:
            time.sleep(3)

            feedback_section = self.driver.find_element(By.CSS_SELECTOR, ".course-feedback")

            feedback_data = {
                "summary": {},
                "reviews": []
            }

            try:
                summary = feedback_section.find_element(By.CSS_SELECTOR, "#course-feedback-summary")
                feedback_data["summary"]["text"] = summary.text

                try:
                    ranks = summary.find_elements(By.CSS_SELECTOR, ".course-rank")
                    ratings = {}
                    for rank in ranks:
                        title = rank.find_element(By.CSS_SELECTOR, ".course-rank-title").text
                        icons = rank.find_element(By.CSS_SELECTOR, ".course-rank-icons")
                        filled_icons = icons.find_elements(By.CSS_SELECTOR, ".fas.fa-star, .fas.fa-weight-hanging")
                        half_icons = icons.find_elements(By.CSS_SELECTOR, ".fas.fa-star-half-alt")
                        rating = len(filled_icons) + (len(half_icons) * 0.5)
                        ratings[title] = rating
                    feedback_data["summary"]["ratings"] = ratings
                except:
                    pass

            except NoSuchElementException:
                feedback_data["summary"]["text"] = "No summary available"

            try:
                carousel = feedback_section.find_element(By.CSS_SELECTOR, "#course-feedback-carousel")
                carousel_items = carousel.find_elements(By.CSS_SELECTOR, ".carousel-item")
                total_reviews = len(carousel_items)
                self.log_debug(f"Found {total_reviews} reviews")

                if total_reviews > 0:
                    first_review = self.extract_single_review(carousel_items[0])
                    if first_review:
                        feedback_data["reviews"].append(first_review)

                    for i in range(1, total_reviews):
                        try:
                            next_button = carousel.find_element(By.CSS_SELECTOR, ".carousel-control-next")
                            next_button.click()
                            time.sleep(0.8)

                            active_item = carousel.find_element(By.CSS_SELECTOR, ".carousel-item.active")
                            review = self.extract_single_review(active_item)
                            if review:
                                feedback_data["reviews"].append(review)

                        except Exception as e:
                            self.log_debug(f"Error extracting review {i}: {e}")
                            continue

            except NoSuchElementException:
                self.log_debug("No carousel found")
                pass

            self.log_debug(f"✓ Extracted {len(feedback_data['reviews'])} reviews")
            return feedback_data

        except Exception as e:
            self.log_debug(f"✗ ERROR extracting feedback: {e}")
            return {"summary": {}, "reviews": []}

    def extract_single_review(self, carousel_item):
        """Extract data from a single carousel item"""
        try:
            review = {}

            try:
                semester = carousel_item.find_element(By.CSS_SELECTOR, ".box-title").text
                review["semester"] = semester
            except:
                review["semester"] = "Unknown"

            try:
                content = carousel_item.find_element(By.CSS_SELECTOR, ".box-content")
                review["content"] = content.text
            except:
                review["content"] = ""

            try:
                ranks = carousel_item.find_elements(By.CSS_SELECTOR, ".course-rank")
                ratings = {}
                for rank in ranks:
                    title = rank.find_element(By.CSS_SELECTOR, ".course-rank-title").text
                    icons = rank.find_element(By.CSS_SELECTOR, ".course-rank-icons")
                    filled_icons = icons.find_elements(By.CSS_SELECTOR, ".fas.fa-star, .fas.fa-weight-hanging")
                    half_icons = icons.find_elements(By.CSS_SELECTOR, ".fas.fa-star-half-alt")
                    rating = len(filled_icons) + (len(half_icons) * 0.5)
                    ratings[title] = rating
                review["ratings"] = ratings
            except:
                review["ratings"] = {}

            return review

        except Exception:
            return None

    def extract_histograms(self):
        """Extract histogram data"""
        self.log_debug("Extracting histograms...")
        try:
            time.sleep(2)

            histogram_section = self.driver.find_element(By.CSS_SELECTOR, ".inline-histograms")

            histogram_data = {
                "raw_text": histogram_section.text
            }

            self.log_debug("✓ Extracted histogram data")
            return histogram_data

        except NoSuchElementException:
            self.log_debug("No histogram section found")
            return {}
        except Exception as e:
            self.log_debug(f"✗ ERROR extracting histograms: {e}")
            return {}

    def close_dialog(self):
        """Close the info dialog"""
        self.log_debug("Closing dialog...")
        try:
            # Try to find and click close button
            close_button = self.driver.find_element(By.CSS_SELECTOR, ".bootstrap-dialog .close")
            close_button.click()
            time.sleep(0.5)
            self.log_debug("✓ Dialog closed")
            return True
        except:
            try:
                # Try ESC key
                from selenium.webdriver.common.keys import Keys
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.send_keys(Keys.ESCAPE)
                time.sleep(0.5)
                self.log_debug("✓ Dialog closed with ESC")
                return True
            except Exception as e:
                self.log_debug(f"Could not close dialog: {e}")
                return False

    def scrape_course(self, course_id):
        """Main method to scrape all course data"""
        self.log_debug(f"\n{'='*50}")
        self.log_debug(f"SCRAPING COURSE: {course_id}")
        self.log_debug(f"{'='*50}")

        # Load the page
        self.load_page()

        # Search and add course
        if not self.search_and_add_course(course_id):
            self.log_debug(f"✗ FAILED: Could not add course {course_id}")
            return None

        # Click info button
        if not self.click_info_button(course_id):
            self.log_debug(f"✗ FAILED: Could not click info button for course {course_id}")
            return None

        # Extract all data
        course_data = {
            "course_id": str(course_id),
            "scraped_at": datetime.now().isoformat(),
            "information": self.extract_course_information(),
            "feedback": self.extract_feedback(),
            "histograms": self.extract_histograms()
        }

        self.log_debug(f"✓ Successfully scraped all data for course {course_id}")
        return course_data

    def save_to_json(self, data, output_dir):
        """Save data to JSON file in specified directory"""
        filename = f"course_{data['course_id']}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.log_debug(f"✓ Saved to: {filepath}")
        return filepath


class BatchScraper:
    """Handles batch scraping with course discovery, recovery, and progress tracking"""

    def __init__(self, output_dir="scraped_courses", progress_file="scraping_progress.json"):
        self.output_dir = output_dir
        self.progress_file = progress_file
        self.discovered_courses_file = "discovered_courses.json"
        self.scraped_courses = set()
        self.failed_courses = set()
        self.all_courses = []

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load progress if exists
        self.load_progress()

    def load_progress(self):
        """Load previously scraped courses from progress file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.scraped_courses = set(data.get('scraped', []))
                    self.failed_courses = set(data.get('failed', []))
                print(f"Loaded progress: {len(self.scraped_courses)} courses already scraped")
                if self.failed_courses:
                    print(f"  {len(self.failed_courses)} courses previously failed")
                    print(f"  Failed courses: {sorted(list(self.failed_courses))}")
            except Exception as e:
                print(f"Could not load progress file: {e}")

    def save_progress(self):
        """Save current progress to file"""
        data = {
            'scraped': list(self.scraped_courses),
            'failed': list(self.failed_courses),
            'last_updated': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_discovered_courses(self):
        """Load previously discovered courses"""
        if os.path.exists(self.discovered_courses_file):
            try:
                with open(self.discovered_courses_file, 'r') as f:
                    self.all_courses = json.load(f)
                print(f"Loaded {len(self.all_courses)} previously discovered courses")
                return True
            except Exception as e:
                print(f"Could not load discovered courses: {e}")
                return False
        return False

    def save_discovered_courses(self):
        """Save discovered courses to file"""
        with open(self.discovered_courses_file, 'w') as f:
            json.dump(self.all_courses, f, indent=2)
        print(f"Saved {len(self.all_courses)} discovered courses to {self.discovered_courses_file}")

    def discover_all_courses(self, prefixes):
        """Discover all courses for given prefixes"""
        print("\n" + "="*60)
        print("PHASE 1: DISCOVERING COURSES")
        print("="*60)

        with CheeseforkScraper(headless=True, debug=False) as scraper:
            scraper.load_page()

            all_discovered = []

            for prefix in tqdm(prefixes, desc="Discovering by prefix"):
                courses = scraper.discover_courses_by_prefix(prefix)
                all_discovered.extend(courses)
                time.sleep(1)  # Be polite to the server

            self.all_courses = all_discovered
            self.save_discovered_courses()

        print(f"\nTotal courses discovered: {len(self.all_courses)}")
        return self.all_courses

    def scrape_all_courses(self, retry_failed=False, debug=True):
        """Scrape all discovered courses"""
        if not self.all_courses:
            print("No courses to scrape. Run discovery first.")
            return

        print("\n" + "="*60)
        print("PHASE 2: SCRAPING COURSES")
        print("="*60)

        # Filter to remaining courses
        if retry_failed:
            courses_to_scrape = [c for c in self.all_courses
                                if c['course_id'] not in self.scraped_courses]
        else:
            courses_to_scrape = [c for c in self.all_courses
                                if c['course_id'] not in self.scraped_courses
                                and c['course_id'] not in self.failed_courses]

        print(f"\nTotal courses discovered: {len(self.all_courses)}")
        print(f"Already scraped: {len(self.scraped_courses)}")
        print(f"Previously failed: {len(self.failed_courses)}")
        print(f"To scrape now: {len(courses_to_scrape)}")

        if not courses_to_scrape:
            print("\nNo courses to scrape!")
            return

        print("\nStarting scraping process...\n")

        # Scrape with progress bar - create new browser for EACH course
        for i, course_info in enumerate(courses_to_scrape):
            course_id = course_info['course_id']

            print(f"\n{'='*60}")
            print(f"Course {i+1}/{len(courses_to_scrape)}: {course_id} - {course_info.get('name', 'Unknown')}")
            print(f"{'='*60}")

            try:
                # Create a COMPLETELY fresh browser session for each course
                print(f"Creating new browser session for course {course_id}...")
                with CheeseforkScraper(headless=True, debug=debug) as scraper:
                    course_data = scraper.scrape_course(course_id)

                    if course_data:
                        # Save to JSON
                        scraper.save_to_json(course_data, self.output_dir)

                        # Mark as scraped
                        self.scraped_courses.add(course_id)

                        # Remove from failed if it was there
                        self.failed_courses.discard(course_id)

                        print(f"✓ SUCCESS: Course {course_id} scraped and saved")
                    else:
                        # Mark as failed
                        self.failed_courses.add(course_id)
                        print(f"✗ FAILED: Course {course_id} returned no data")

                # Save progress after each course
                self.save_progress()

                # Small delay between courses
                time.sleep(2)

            except KeyboardInterrupt:
                print("\n\nScraping interrupted by user. Progress has been saved.")
                self.save_progress()
                raise
            except Exception as e:
                print(f"\n✗ EXCEPTION scraping course {course_id}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.failed_courses.add(course_id)
                self.save_progress()
                continue

        # Final summary
        print("\n" + "="*60)
        print("SCRAPING COMPLETED")
        print("="*60)
        print(f"Successfully scraped: {len(self.scraped_courses)} courses")
        print(f"Failed: {len(self.failed_courses)} courses")
        print(f"Output directory: {self.output_dir}")
        print(f"Progress file: {self.progress_file}")

        if self.failed_courses:
            print(f"\nFailed course IDs: {sorted(list(self.failed_courses))}")

    def run_full_scrape(self, prefixes, skip_discovery=False, retry_failed=False, debug=True):
        """Run the complete scraping process"""
        # Phase 1: Discover courses (or load from file)
        if skip_discovery and self.load_discovered_courses():
            print("Using previously discovered courses")
        else:
            self.discover_all_courses(prefixes)

        # Phase 2: Scrape courses
        self.scrape_all_courses(retry_failed=retry_failed, debug=debug)


def main():
    """Main execution function"""
    print("="*60)
    print("Cheesefork Batch Course Scraper")
    print("="*60)

    # Define course prefixes to scrape
    prefixes = ['094', '095', '096', '097', '098']

    # Create batch scraper
    batch_scraper = BatchScraper(
        output_dir="scraped_courses_spring_2026",
        progress_file="scraping_progress.json"
    )

    # Check if we should skip discovery
    skip_discovery = os.path.exists(batch_scraper.discovered_courses_file)
    if skip_discovery:
        response = input("\nFound previously discovered courses. Skip discovery phase? (y/n): ")
        skip_discovery = response.lower() == 'y'

    # Ask about retrying failed courses
    if batch_scraper.failed_courses:
        response = input(f"\nFound {len(batch_scraper.failed_courses)} previously failed courses. Retry them? (y/n): ")
        retry_failed = response.lower() == 'y'
    else:
        retry_failed = False

    # Start scraping
    try:
        batch_scraper.run_full_scrape(
            prefixes=prefixes,
            skip_discovery=skip_discovery,
            retry_failed=retry_failed,
            debug=True  # Enable debug logging
        )
    except KeyboardInterrupt:
        print("\n\nGracefully shutting down...")


if __name__ == "__main__":
    main()