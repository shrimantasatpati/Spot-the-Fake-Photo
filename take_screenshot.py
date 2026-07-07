"""
Take a screenshot of the Streamlit app by uploading a real image and a screen image,
then capturing the results.
"""
import base64
import time
from playwright.sync_api import sync_playwright

def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def screenshot(page, out_path):
    page.screenshot(path=out_path, full_page=False)
    print("Saved: " + out_path)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        
        # Go to Streamlit app
        page.goto("http://localhost:8501")
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Upload a real image via the file uploader
        real_img = "dataset/real/20260706_154141.jpg"
        page.set_input_files('input[type="file"]', real_img)
        time.sleep(4)  # Wait for prediction
        
        screenshot(page, "final_demo_screenshot.png")
        browser.close()

if __name__ == "__main__":
    main()
