const puppeteer = require('puppeteer');
const fs = require('fs');

async function crawlRedditSubreddit(subredditUrl, numPagesToCrawl = 2) {
    let browser;
    const allPostTexts = new Set(); // Use a Set to store unique texts

    try {
        browser = await puppeteer.launch({
            headless: false, // Set to false to see the browser GUI
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"'
            ]
        });
        const page = await browser.newPage();
        // Set a reasonable viewport size
        await page.setViewport({ width: 1200, height: 800 });

        console.log(`Navigating to: ${subredditUrl}`);
        await page.goto(subredditUrl, { waitUntil: 'domcontentloaded' });

        // Wait for the initial content to load
        await page.waitForSelector('article [slot="text-body"]', { timeout: 20000 });
        console.log("Initial page loaded.");

        for (let i = 0; i < numPagesToCrawl; i++) {
            // Scroll down to load more content
            console.log("Scrolling down to load more content...");
            await page.evaluate(() => {
                window.scrollTo(0, document.body.scrollHeight);
            });
            await new Promise(resolve => setTimeout(resolve, 3000)); // Wait for content to load

            // Try to click a "load more" button if present
            try {
                // These selectors might need adjustment based on Reddit's UI changes
                const loadMoreButtonSelector = 'button[data-testid="load-more-button"], a[role="button"][href*="/next"]';
                const loadMoreButton = await page.waitForSelector(loadMoreButtonSelector, { timeout: 5000, visible: true });
                
                if (loadMoreButton) {
                    console.log("Clicking 'Load More' button...");
                    await loadMoreButton.click();
                    await new Promise(resolve => setTimeout(resolve, 3000)); // Wait for new content to load
                } else {
                    console.log("No 'Load More' button found or not visible.");
                }
            } catch (error) {
                console.log("No 'Load More' button found or clickable on this scroll iteration.");
            }

            // Extract text from visible articles
            console.log("Extracting post texts...");
            const currentPostTexts = await page.evaluate(() => {
                const textElements = Array.from(document.querySelectorAll('article [slot="text-body"]'));
                return textElements.map(row => row.innerText).filter(text => text.trim() !== '');
            });

            let newTextsCount = 0;
            currentPostTexts.forEach(text => {
                if (!allPostTexts.has(text)) {
                    allPostTexts.add(text);
                    newTextsCount++;
                }
            });
            console.log(`Found ${newTextsCount} new posts on this iteration.`);

            if (newTextsCount === 0 && i > 0) { // Break if no new posts were found after first iteration
                console.log("No new posts found after scrolling/clicking. Assuming end of content.");
                break;
            }
        }

    } catch (error) {
        console.error(`An error occurred: ${error.message}`);
    } finally {
        if (browser) {
            await browser.close();
            console.log("Browser closed.");
        }
    }

    return Array.from(allPostTexts); // Convert Set to Array before returning
}

// Main execution
(async () => {
    const subredditUrl = 'https://www.reddit.com/r/smallbusiness/';
    const extractedTexts = await crawlRedditSubreddit(subredditUrl, 5); // Crawl 5 "pages"

    console.log("\n--- Extracted Post Texts ---");
    extractedTexts.slice(0, 10).forEach((text, index) => { // Print first 10 for demonstration
        console.log(`--- Post ${index + 1} ---`);
        console.log(text);
        console.log("-".repeat(20));
    });

    if (extractedTexts.length > 10) {
        console.log(`\n... ${extractedTexts.length - 10} more posts not displayed.`);
    }

    // You can save this data to a file
    fs.writeFileSync('data/input/reddit_smallbusiness_posts.json', JSON.stringify(extractedTexts, null, 2), 'utf-8');
})();