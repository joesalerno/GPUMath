
from playwright.sync_api import sync_playwright

def verify(page):
    page.goto('http://localhost:5173')
    page.wait_for_selector('#gpuCanvas')
    # Wait longer for potential software rendering compilation
    page.wait_for_timeout(5000)
    page.screenshot(path='verification/fractal_render.png')
    print('Screenshot captured')

with sync_playwright() as p:
    # enable-unsafe-webgpu is required.
    # use-gl=swiftshader might help if no hardware GPU is present,
    # but WebGPU usually requires vulkan on Linux.
    # We try with the unsafe flag first.
    browser = p.chromium.launch(
        headless=True,
        args=['--enable-unsafe-webgpu', '--use-gl=swiftshader']
    )
    page = browser.new_page()
    try:
        verify(page)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        browser.close()
