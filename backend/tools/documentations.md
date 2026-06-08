# Data Source API Documentation & Verification

Verified against official documentation on 2026-06-06.

---

## Exchange APIs

### OKX

| Endpoint | Doc URL |
|----------|---------|
| Get Ticker | https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-ticker |
| Get Candlesticks | https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks |
| Get Order Book | https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-order-book |
| Get Trades | https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-trades |

**Status:** All endpoints correct. URL paths, parameters, response fields, and limits all match.
Minor: module docstring references `/api/v5/public/` but `BASE_URL` correctly uses `/api/v5`.

### Binance

| Endpoint | Doc URL |
|----------|---------|
| 24hr Ticker | https://binance-docs.github.io/apidocs/spot/en/#24hr-ticker-price-change-statistics |
| Kline/Candlestick | https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data |
| Order Book | https://binance-docs.github.io/apidocs/spot/en/#order-book |
| Recent Trades | https://binance-docs.github.io/apidocs/spot/en/#recent-trades-list |
| REST API (raw) | https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md |

**Status:** All endpoints correct. `fetch_orderbook` limit updated to 5000 to match current API spec (was capped at 1000).

### CoinGecko

| Endpoint | Doc URL |
|----------|---------|
| Simple Price | https://docs.coingecko.com/reference/simple-price |
| Market Chart | https://docs.coingecko.com/reference/coins-id-market-chart |
| Trending | https://docs.coingecko.com/reference/trending-search |

**Status:** All endpoints correct. URL paths, parameters, and response parsing all match.

### Alpha Vantage

| Endpoint | Doc URL |
|----------|---------|
| All endpoints | https://www.alphavantage.co/documentation/ |

**Status:** All three endpoints (`GLOBAL_QUOTE`, `TIME_SERIES_DAILY`, `CURRENCY_EXCHANGE_RATE`) correct. Response field names including numbered prefixes (`"01. symbol"`, `"1. open"`, etc.) all verified.

### Polygon.io

| Endpoint | Doc URL |
|----------|---------|
| Ticker Details | https://polygon.io/docs/stocks/get_v3_reference_tickers__ticker |
| Aggregates | https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to |
| Last NBBO Quote | https://polygon.io/docs/stocks/get_v2_last_nbbo__stocksticker |

**Status:** All endpoints correct. `fetch_last_quote` bid/ask field mapping fixed (was swapped: `P`→ask, `p`→bid per Polygon docs).

### Finnhub

| Endpoint | Doc URL |
|----------|---------|
| Quote | https://finnhub.io/docs/api/quote |
| Company News | https://finnhub.io/docs/api/company-news |
| Earnings | https://finnhub.io/docs/api/company-earnings |

**Status:** All endpoints correct. `from_date` and `to_date` on `fetch_company_news` are now required parameters (matching the API spec).

---

## Macro APIs

### FRED

| Endpoint | Doc URL |
|----------|---------|
| Series Observations | https://fred.stlouisfed.org/docs/api/fred/series_observations.html |
| Series Info | https://fred.stlouisfed.org/docs/api/fred/series.html |
| Series Search | https://fred.stlouisfed.org/docs/api/fred/series_search.html |
| Main index | https://fred.stlouisfed.org/docs/api/fred/ |

**Status:** All endpoints correct. `api_key` is now a required parameter with no default (was `"DEMO_KEY"` which doesn't work). Register at https://fredaccount.stlouisfed.org for a free key.

### ECB

| Endpoint | Doc URL |
|----------|---------|
| API Overview | https://data.ecb.europa.eu/help/api/overview |
| Data Retrieval | https://data.ecb.europa.eu/help/api/data |

**Status:** All endpoints correct. `rate_type` docstring fixed: valid codes are `MRR_FR` (main refinancing, fixed rate), `DFR` (deposit facility), `MLFR` (marginal lending). Invalid codes `MRO` and `MLF` removed from documentation.

### Nasdaq Data Link (Quandl)

| Endpoint | Doc URL |
|----------|---------|
| In-depth Usage | https://docs.data.nasdaq.com/docs/in-depth-usage |

**Status:** All endpoints correct. URL paths (`/datasets/{db}/{ds}/data.json`, `/datasets/{db}/{ds}/metadata.json`), parameters, and response wrappers (`dataset_data`, `dataset`) all match.

---

## News APIs

### The Guardian

| Endpoint | Doc URL |
|----------|---------|
| Open Platform Docs | https://open-platform.theguardian.com/documentation/ |

**Status:** All endpoints correct. `api-key="test"` is a valid free-tier development key (heavily rate-limited). All parameter names (hyphenated: `api-key`, `page-size`, `order-by`, `show-fields`) and response fields (`response.results[].webTitle`, etc.) verified.

### Hacker News

| Endpoint | Doc URL |
|----------|---------|
| API Reference | https://github.com/HackerNews/API |

**Status:** All endpoints correct. `/topstories.json` returns up to 500 IDs. `/item/{id}.json` fields (`id`, `type`, `title`, `url`, `text`, `score`, `by`, `time`, `descendants`) all verified.

### NewsAPI

| Endpoint | Doc URL |
|----------|---------|
| Top Headlines | https://newsapi.org/docs/endpoints/top-headlines |
| Everything | https://newsapi.org/docs/endpoints/everything |

**Status:** All endpoints correct. Parameter names (`apiKey`, `country`, `pageSize`, `q`, `category`, `sortBy`, `language`) and response fields (`articles[].title`, `source.name`, `publishedAt`, etc.) all match.

---

## Social APIs

### Mastodon

| Endpoint | Doc URL |
|----------|---------|
| Timelines | https://docs.joinmastodon.org/methods/timelines/ |
| Search | https://docs.joinmastodon.org/methods/search/ |
| Status entity | https://docs.joinmastodon.org/entities/Status/ |

**Status:** All endpoints correct. Public timeline at `/api/v1/timelines/public`, hashtag at `/api/v1/timelines/tag/{tag}`, search at `/api/v2/search` — all verified. Response fields match.

### Twitter/X

| Endpoint | Doc URL |
|----------|---------|
| Recent Search | https://docs.x.com/x-api/posts/recent-search |
| User Tweets | https://docs.x.com/x-api/posts/user-posts-timeline-by-user-id |
| User Lookup | https://docs.x.com/x-api/users/user-lookup-by-username |
| Fields reference | https://docs.x.com/x-api/fundamentals/fields |

**Status:** All endpoints functionally correct. Minor: base URL uses legacy `api.twitter.com` (docs now reference `api.x.com`); both domains resolve identically.
