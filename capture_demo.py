import base64
import os
import subprocess
import time
from playwright.sync_api import sync_playwright

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def main():
    # Start the Flask app in the background
    print("Starting Flask app...")
    flask_process = subprocess.Popen(["python", "demo_web/app.py"])
    
    # Wait for the app to start
    time.sleep(5)
    
    # Path to one of the real dataset images
    real_images_dir = 'dataset/real'
    first_image = os.path.join(real_images_dir, os.listdir(real_images_dir)[0])
    base64_img = get_base64_image(first_image)
    
    print("Capturing screenshot...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to the local server
        page.goto("http://127.0.0.1:7860/")
        
        # Inject the image by replacing the video element
        js_code = f"""
        const video = document.getElementById('video');
        const img = document.createElement('img');
        img.id = 'video';
        img.width = 640;
        img.height = 480;
        img.src = 'data:image/jpeg;base64,{base64_img}';
        video.parentNode.replaceChild(img, video);
        """
        page.evaluate(js_code)
        
        # Wait for the UI to update the prediction (usually takes 500ms)
        time.sleep(3)
        
        # Take screenshot
        page.screenshot(path="final_demo_screenshot.png", full_page=True)
        print("Saved screenshot to final_demo_screenshot.png")
        
        browser.close()
        
    print("Shutting down Flask app...")
    flask_process.terminate()

if __name__ == '__main__':
    main()
