from playwright.sync_api import sync_playwright

def verify_fractal_load():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()
        try:
            page.goto("http://localhost:5173")

            # Wait for canvas
            page.wait_for_selector("#gpuCanvas")

            # Wait a bit for shader compilation and rendering
            page.wait_for_timeout(3000)

            # Take screenshot
            page.screenshot(path="verification/fractal_loaded.png")
            print("Screenshot taken.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_fractal_load()
