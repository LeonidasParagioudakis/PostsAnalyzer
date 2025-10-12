const puppeteer = require('puppeteer');
const fs = require('fs');

function parseArgs() {
    const parsedArgs = {};
    const args = process.argv.slice(2);
    if (args.length === 0) {
        return;
    }
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if (arg.startsWith('--')) {
            const key = arg.slice(2);
            const value = args[i + 1];
            if (value && !value.startsWith('--')) {
                parsedArgs[key] = value;
                i++;
            } else {
                parsedArgs[key] = true;
            }
        }
    }
    return parsedArgs;
}

function calculateSavePath(filename) {
    const safeName = filename.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    return `data_${safeName}_${Date.now()}.json`;
}

async function autoScroll(page) {
    await page.evaluate(async () => {
        await new Promise((resolve) => {
            let totalHeight = 0;
            const distance = 200;
            const timer = setInterval(() => {
                const scrollHeight = document.body.scrollHeight;
                window.scrollBy(0, distance);
                totalHeight += distance;
                if (totalHeight >= scrollHeight) {
                    clearInterval(timer);
                    resolve();
                }
            }, 200);
        });
    });
}

async function crawlCategory(page, categoryUrl) {
    let productLinks = [];
    let hasNext = true;

    while (hasNext) {
        console.log(`Visiting category page: ${categoryUrl}`);
        await page.goto(categoryUrl, { waitUntil: 'networkidle2' });
        await autoScroll(page);

        const links = await page.$$eval('a.js-sku-link', els => els.map(e => e.href));
        productLinks.push(...links);

        const nextButton = await page.$('a.next_page');
        if (nextButton) {
            categoryUrl = await page.$eval('a.next_page', el => el.href);
        } else {
            hasNext = false;
        }
    }

    return [...new Set(productLinks)];
}

async function crawlProductComments(page, productUrl) {
    console.log(`  Visiting product: ${productUrl}`);
    await page.goto(productUrl, { waitUntil: 'networkidle2' });
    await autoScroll(page);

    // Click "See all reviews" if exists
    const seeAllSelector = 'a.reviews_link';
    if (await page.$(seeAllSelector)) {
        await Promise.all([
            page.click(seeAllSelector),
            page.waitForNavigation({ waitUntil: 'networkidle2' })
        ]);
        await autoScroll(page);
    }

    let comments = [];
    let hasNextComments = true;

    while (hasNextComments) {
        const pageComments = await page.$$eval('.review-content', els =>
            els.map(e => e.innerText.trim())
        );
        comments.push(...pageComments);

        const nextCommentsBtn = await page.$('a.next_page');
        if (nextCommentsBtn) {
            const nextHref = await page.$eval('a.next_page', el => el.href);
            await page.goto(nextHref, { waitUntil: 'networkidle2' });
            await autoScroll(page);
        } else {
            hasNextComments = false;
        }
    }

    return comments;
}

(async () => {
    const args = parseArgs();
    if (!args?.urls) {
        console.log("Use script like this: node skroutz_gr.js --urls category_urls.txt");
        return;
    }

    const urls = fs.readFileSync(args.urls, 'utf-8')
        .split('\n')
        .map(l => l.trim())
        .filter(Boolean);

    const browser = await puppeteer.launch({
        headless: false, // Set to false to see the browser GUI
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"'
        ]
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 800 });

    let results = [];

    for (const categoryUrl of urls) {
        const productLinks = await crawlCategory(page, categoryUrl);

        for (const productUrl of productLinks) {
            const comments = await crawlProductComments(page, productUrl);
            results.push({
                productUrl,
                comments
            });
        }
    }

    await browser.close();

    const savePath = calculateSavePath('skroutz_comments');
    fs.writeFileSync(savePath, JSON.stringify(results, null, 2), 'utf-8');
    console.log(`Saved ${results.length} products with comments to ${savePath}`);
})();
