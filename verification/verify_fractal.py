from playwright.sync_api import sync_playwright

def verify(page):
    page.goto("http://localhost:5173")
    page.wait_for_selector("#gpuCanvas")

    # Wait a bit for the GPU to render (simulated)
    page.wait_for_timeout(2000)

    # Check if UI elements are present
    assert page.is_visible("#maxIterRange")
    assert page.is_visible("#maxIterVal")

    # Take screenshot
    page.screenshot(path="verification/fractal_ui.png")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    verify(page)
    browser.close()
