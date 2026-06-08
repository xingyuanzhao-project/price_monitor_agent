"""
Registry of supported public and additional data sources.

Single source of truth for what data sources appear in the Settings UI
and what the tools can connect to at runtime. Every source listed here
MUST have a corresponding support module in backend/tools/supports/.

Categories (fixed order): macro, exchange, news, social
"""

from dataclasses import dataclass


CATEGORY_ORDER = ["macro", "exchange", "news", "social"]


@dataclass
class PublicDataSource:
    """A free/open data source that requires no API key."""
    source_id: str
    name: str
    category: str
    base_url: str
    description: str = ""


@dataclass
class AdditionalApiSource:
    """A data source that requires a user-provided API key or endpoint."""
    source_id: str
    name: str
    category: str
    default_base_url: str = ""
    description: str = ""


PUBLIC_DATA_SOURCES: list[PublicDataSource] = [
    # --- macro ---
    PublicDataSource(
        source_id="fred",
        name="FRED",
        category="macro",
        base_url="https://api.stlouisfed.org/fred/",
        description="Federal Reserve Economic Data (US macro indicators)",
    ),
    PublicDataSource(
        source_id="ecb",
        name="ECB",
        category="macro",
        base_url="https://data-api.ecb.europa.eu/service/data/",
        description="European Central Bank statistical data",
    ),
    # --- exchange ---
    PublicDataSource(
        source_id="okx",
        name="OKX",
        category="exchange",
        base_url="https://www.okx.com/api/v5/public/",
        description="OKX public market data (tickers, candles, orderbook, trades)",
    ),
    PublicDataSource(
        source_id="binance",
        name="Binance",
        category="exchange",
        base_url="https://api.binance.com/api/v3/",
        description="Binance public market data (tickers, candles, depth, trades)",
    ),
    PublicDataSource(
        source_id="coingecko",
        name="CoinGecko",
        category="exchange",
        base_url="https://api.coingecko.com/api/v3/",
        description="CoinGecko aggregated crypto market data",
    ),
    # --- news ---
    PublicDataSource(
        source_id="guardian",
        name="The Guardian",
        category="news",
        base_url="https://content.guardianapis.com/",
        description="The Guardian open platform (headlines, articles, search)",
    ),
    PublicDataSource(
        source_id="hackernews",
        name="Hacker News",
        category="news",
        base_url="https://hacker-news.firebaseio.com/v0/",
        description="Hacker News public API (top/new/best stories, comments)",
    ),
    # --- social ---
    PublicDataSource(
        source_id="mastodon",
        name="Mastodon",
        category="social",
        base_url="https://mastodon.social/api/v1/",
        description="Mastodon public timeline and account data",
    ),
    # --- exchange (added) ---
    PublicDataSource(
        source_id="yahoo",
        name="Yahoo Finance",
        category="exchange",
        base_url="https://query1.finance.yahoo.com/v8/finance/chart/",
        description="Stocks, ETFs, indices via Yahoo Finance unofficial chart API",
    ),
    PublicDataSource(
        source_id="frankfurter",
        name="Frankfurter",
        category="exchange",
        base_url="https://api.frankfurter.dev/v1/",
        description="FX rates from 84 central banks, 201 currencies, back to 1948",
    ),
    # --- macro (added) ---
    PublicDataSource(
        source_id="worldbank",
        name="World Bank",
        category="macro",
        base_url="https://api.worldbank.org/v2/",
        description="1600+ development indicators for 200+ countries",
    ),
    PublicDataSource(
        source_id="imf",
        name="IMF DataMapper",
        category="macro",
        base_url="https://www.imf.org/external/datamapper/api/v1/",
        description="Macroeconomic forecasts and indicators for ~190 countries",
    ),
    # --- news (added) ---
    PublicDataSource(
        source_id="gdelt",
        name="GDELT",
        category="news",
        base_url="https://api.gdeltproject.org/api/v2/",
        description="Global news events from 100+ languages, updated every 15 minutes",
    ),
    PublicDataSource(
        source_id="isw",
        name="ISW",
        category="news",
        base_url="https://understandingwar.org/",
        description="Military and geopolitical assessments from Institute for the Study of War",
    ),
    PublicDataSource(
        source_id="oksurf",
        name="OKSURF",
        category="news",
        base_url="https://ok.surf/api/v1/",
        description="Google News headlines by section (Business, Tech, World, etc.)",
    ),
    PublicDataSource(
        source_id="wikievents",
        name="WikiEvents",
        category="news",
        base_url="https://offstream.news/",
        description="Wikipedia Current Events in structured JSON",
    ),
    PublicDataSource(
        source_id="thehear",
        name="The Hear",
        category="news",
        base_url="https://www.thehear.org/api/",
        description="Multi-country headline aggregator with AI overviews, 20 countries",
    ),
    # --- social (added) ---
    PublicDataSource(
        source_id="github",
        name="GitHub",
        category="social",
        base_url="https://api.github.com/",
        description="Trending repos and developer community activity",
    ),
    PublicDataSource(
        source_id="lemmy",
        name="Lemmy",
        category="social",
        base_url="https://lemmy.ml/api/v3/",
        description="Federated forums -- crypto, finance, wallstreetbets communities",
    ),
    # --- prediction markets ---
    PublicDataSource(
        source_id="polymarket",
        name="Polymarket",
        category="exchange",
        base_url="https://gamma-api.polymarket.com/",
        description="Prediction market probabilities on geopolitical and economic events",
    ),
    PublicDataSource(
        source_id="predscope",
        name="PredScope",
        category="exchange",
        base_url="https://predscope.com/api/",
        description="Aggregated Polymarket data -- top 100 markets with probabilities",
    ),
    # --- weather/climate ---
    PublicDataSource(
        source_id="openmeteo",
        name="Open-Meteo",
        category="macro",
        base_url="https://api.open-meteo.com/v1/",
        description="Weather forecasts and ERA5 climate reanalysis, global coverage",
    ),
    # --- macro (added) ---
    PublicDataSource(
        source_id="bis",
        name="BIS Statistics",
        category="macro",
        base_url="https://stats.bis.org/api/v1/",
        description="Central bank policy rates, REER, credit-to-GDP for 40+ economies",
    ),
    PublicDataSource(
        source_id="usaspending",
        name="USA Spending",
        category="macro",
        base_url="https://api.usaspending.gov/api/v2/",
        description="US federal spending and contract data",
    ),
    PublicDataSource(
        source_id="comtrade",
        name="UN Comtrade",
        category="macro",
        base_url="https://comtradeapi.un.org/public/v1/",
        description="Global merchandise trade flows between 200+ countries",
    ),
    # --- news/disaster ---
    PublicDataSource(
        source_id="usgs",
        name="USGS Earthquake",
        category="news",
        base_url="https://earthquake.usgs.gov/earthquakes/feed/v1.0/",
        description="Real-time M4.5+ earthquakes globally, updated every 5 minutes",
    ),
    PublicDataSource(
        source_id="gdacs",
        name="GDACS",
        category="news",
        base_url="https://www.gdacs.org/gdacsapi/api/",
        description="UN disaster alerts -- earthquakes, floods, cyclones, volcanoes",
    ),
    PublicDataSource(
        source_id="eonet",
        name="NASA EONET",
        category="news",
        base_url="https://eonet.gsfc.nasa.gov/api/v3/",
        description="Natural events -- wildfires, severe storms, volcanoes, floods",
    ),
]

ADDITIONAL_API_SOURCES: list[AdditionalApiSource] = [
    # --- macro ---
    AdditionalApiSource(
        source_id="quandl",
        name="Nasdaq Data Link",
        category="macro",
        default_base_url="https://data.nasdaq.com/api/v3/",
        description="Economic, financial, and alternative datasets",
    ),
    # --- exchange ---
    AdditionalApiSource(
        source_id="alphavantage",
        name="Alpha Vantage",
        category="exchange",
        default_base_url="https://www.alphavantage.co/query",
        description="Stock, forex, crypto data (requires free API key)",
    ),
    AdditionalApiSource(
        source_id="polygon",
        name="Polygon.io",
        category="exchange",
        default_base_url="https://api.polygon.io/",
        description="US stock, options, forex, crypto data",
    ),
    AdditionalApiSource(
        source_id="finnhub",
        name="Finnhub",
        category="exchange",
        default_base_url="https://finnhub.io/api/v1/",
        description="Stock market data, earnings, filings",
    ),
    # --- news ---
    AdditionalApiSource(
        source_id="newsapi",
        name="NewsAPI",
        category="news",
        default_base_url="https://newsapi.org/v2/",
        description="Global news headlines and articles",
    ),
    # --- social ---
    AdditionalApiSource(
        source_id="twitter",
        name="Twitter/X",
        category="social",
        default_base_url="https://api.twitter.com/2/",
        description="Twitter/X posts, timelines, search (v2 API)",
    ),
]


def get_public_sources_by_category() -> dict[str, list[dict]]:
    """Group public sources by category, in fixed category order."""
    result: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_ORDER}
    for src in PUBLIC_DATA_SOURCES:
        entry = {"source_id": src.source_id, "name": src.name, "base_url": src.base_url, "description": src.description}
        result[src.category].append(entry)
    return result


def get_additional_sources_by_category() -> dict[str, list[dict]]:
    """Group additional API sources by category, in fixed category order."""
    result: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_ORDER}
    for src in ADDITIONAL_API_SOURCES:
        entry = {"source_id": src.source_id, "name": src.name, "default_base_url": src.default_base_url, "description": src.description}
        result[src.category].append(entry)
    return result
