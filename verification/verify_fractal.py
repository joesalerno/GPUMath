from playwright.sync_api import sync_playwright
import time

def verify_fractal_rendering():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        context = browser.new_context()
        page = context.new_page()

        try:
            # Go to localhost
            page.goto("http://localhost:5173")

            # Wait for canvas to be present
            page.wait_for_selector("#gpuCanvas")

            # Wait a bit for the GPU initialization and rendering
            # Since we can't easily hook into the console log here without more setup,
            # we'll just wait a couple of seconds.
            time.sleep(2)

            # Check console logs for errors (we can't see them directly but we can try to fail if needed)
            # Actually, let's just take a screenshot. If it's black or error-colored, we'll know.

            page.screenshot(path="verification/fractal_screenshot.png")
            print("Screenshot taken.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_fractal_rendering()
