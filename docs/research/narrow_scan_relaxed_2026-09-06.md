---
title: "Narrow Scan (Relaxed Rules) — Synthesis, 2026-09-06"
tags:
  - doc/research
  - topic/business-readiness
  - status/current
---

# Narrow Scan (Relaxed Rules) — Synthesis, 2026-09-06

Scope: re-scan of security / intel / finance-data / energy sub-fields under three relaxed constraints (derived data allowed; inbound selling allowed; $1M/yr per product with a second product on the same engine). Eight merged finalists went through two skeptic lenses each (reachability-licence, moat-crowding). Eleven candidates were dropped at merge. This report is the decision-grade output.

Verification caveat carried over from every skeptic: the WebSearch budget was exhausted mid-run, so most refutations rest on direct page fetches; a number of primary pages (CZDS terms, OpenINTEL access, CAISO terms, IMO GISIS, BoJ/NBS, OECD, Crunchbase/RocketReach/ZoomInfo estimates) returned 403/404 and are marked unverified where it matters.

---

## 1. Verdict

Relaxing the three constraints opened exactly one thing the strict scan could not: products where the owner *measures or reconciles* public sources himself (the Shodan/Tiingo model) — a DNS resolver archive, an SEC-derived reconciled security master, an open-AIS entity-resolution ledger. It did **not** open the thing the finalists were pitched on. Six of eight finalists led with "a point-in-time archive nobody can back-fill", and the skeptics falsified or gutted that claim in six of eight cases: CAISO archives its curtailment reports, ERCOT 60-day disclosures are downloadable by date, Danish and Norwegian AIS are historical archives, EIA STEO/WPSR go back to 1983/2011, Eurostat keeps every first-print PDF, SAM.gov exposes full notice history via its management API, USAspending publishes monthly delta files to FY2007, the Wayback Machine holds 96 monthly snapshots of `company_tickers.json`, and BiopharmaWatch already sells an `as_of` PDUFA API. Two finalists (PDUFA Vintage, Macro Release Vintage) were killed outright on that basis. The rest survive only after being repositioned from "archive moat" to "correctness/engineering moat with an archive that becomes valuable in 2-3 years". On the comparable rule, the relaxation also did less than hoped: no finalist has a verified $1M+/yr, <=5-person, no-outbound comparable *in its exact niche*. The closest are adjacent-lane: ipinfo (bootstrapped, crossed $1M ARR 2017-18, headcount at crossing unstated), BuiltWith (1 employee; the $14M figure is unsourced), BioPharmCatalyst (3,000 paying subs; headcount 2-10, and its lane was killed on moat). Every other comparable is "shape verified, revenue inferred or third-party estimate". Blunt summary: the relaxation converts the strict scan's "nothing is allowed" into "six things are allowed, none is proven, and all of them start behind at least one cheap incumbent." The two with real verified channels and clean licences are DNS Wayback and Open Security Master; both are 3-year, not 1-year, paths to $1M.

---

## 2. Ranked candidates

All six survivors passed both lenses without a formal refutation, but every one carries corrections that materially change its pitch. None is "clean" in substance; the ranking is by (a) quality of verified comparable, (b) licence cleanliness of the core, (c) what moat is left after corrections, (d) owner-asset reuse.

### 2.1 DNS Wayback — point-in-time active-DNS + CT archive with IP-as-of projection

**What it is.** Owner-performed daily resolution of every CT-seen hostname plus a custom Tranco universe across A/AAAA/NS/MX/TXT/CNAME/CAA/SOA, content-addressed and append-only, exposing "what did X resolve to on date D", "what changed between D1 and D2", and reverse (IP → historical hostnames). Second projection: IP attribution as-of-date (RIR delegated-extended, RouteViews/RIS MRT, cloud-range JSON). The email-posture lead-list and per-ticker infrastructure-feed projections are **dropped or re-scoped** (see licence).

**Data sources and licence status.**
- Resolutions: owner-generated, redistributable (Shodan model). Clean.
- CT logs: public (RFC 6962). Clean.
- CZDS zone files: ICANN Base Registry Agreement Spec 4 §2.1.5 forbids use to "support any marketing activities to entities other than the user's existing customers" and requires protection against disclosure of the data. Selling DMARC/SPF posture *lead lists* over the zone universe violates this; .com/.net are not in CZDS (Verisign ZFA, equivalent terms). Posture may be sold as per-domain lookup/monitoring on CT/Tranco-sourced domains only. https://newgtlds.icann.org/sites/default/files/agreements/agreement-approved-31jul17-en.html
- Tranco default list includes Cloudflare Radar data under CC BY-NC 4.0; generate a custom list excluding it. https://tranco-list.eu/
- Bulk WHOIS: ARIN refuses bulk to product builders "with no clear benefit to the broader community"; ARIN WhoWas is staff-approved, no high-volume use; RIPE DB terms Art 4.4/4.5 forbid marketing use and redistribution of more than an insubstantial part. IP-as-of must rest on delegated-extended stats, RDAP, MRT dumps and cloud JSON — sufficient for allocation/origin/cloud attribution, not org-name history. https://www.arin.net/reference/research/bulkwhois/ ; https://www.ripe.net/manage-ips-and-asns/db/support/documentation/terms
- Rapid7 Sonar FDNS confirmed login-gated since 10 Feb 2022 (still collected daily). https://sonardata.rapid7.com/sonar.fdns_v2/
- OpenINTEL "non-commercial only" could not be verified.

**Who pays and how they find it.** Incident responders, bug-bounty/pentest solo operators, fraud/anti-abuse engineers, domain investors, OSINT journalists. Found via search intent ("DNS history <domain>"), free per-domain and per-IP timeline pages (ipinfo's documented acquisition playbook: Stack Overflow answers, free unauthenticated API, per-IP SEO pages), PyPI client, and a subfinder/amass provider PR (subfinder 14.4k stars already lists SecurityTrails, DNSDB, WhoisXML, BuiltWith as providers). Card at $29-549/mo.

**Verified comparable.**
- ipinfo.io: bootstrapped, founder full-time 2016 at ~$100K/yr, crossed $1M ARR 2017-18, no outbound sales, 15 people today. Headcount at the $1M crossing not stated. https://saasclub.io/podcast/ipinfo-ben-dowling/ ; https://ipinfo.io/blog/bootstrapping-a-startup
- BuiltWith: 1 full-time employee, "no outbound sales team", $295-995/mo; the "$14M+" is not sourced inside the article (last hard number: $40K/month in 2011). https://www.colinkeeley.com/blog/the-story-of-builtwith-1-employee-14m-arr
- Price ceiling/floor: SecurityTrails $500/$1,500 self-serve with historical DNS (https://securitytrails.com/corp/pricing); ViewDNS $29-549 (https://viewdns.info/api/pricing/).

**Moat (after corrections).** Engineering: polite multi-vantage resolution of 20M+ names × 8 types daily, content-addressed diff, reverse index, as-of/diff semantics at $29-49. Vintage: real only against future entrants, and only after 2-3 years — WhoisXML/DomainTools hold "over a decade" of resolution history, CompleteDNS has NS history since 2002 at $30/mo, dns.coffee has free zone-file history since 2011. The "unoccupied slot" framing is wrong; the actual wedge is cheap full-record-set daily as-of/diff plus the IP-as-of projection, which has no verified incumbent (ipinfo/IPLocate pricing pages show no dated lookup).

**Honest kill risk.** Merklemap (EUR49/mo, 700M+ hostnames, 14 record types, already carries `query_timestamp`) exposing history is a storage-and-index change, not a collection change (https://www.merklemap.com/documentation/dns-records-database). WhoisXML DNS Chronicle already sells 116B+ records at credit pricing. The first two years are a price-and-UX fight against Merklemap/CompleteDNS/WhoisXML — the "developer engine vendors give away" pattern the prior scan warned about. Secondary: resolver-abuse complaints against a solo operator.

**Revenue arithmetic.** $1M/yr ≈ 900 × $49 + 200 × $149 + 30 × $549 across DNS and IP-as-of. Plan $200-400K year one, $1M by year three when vintage is the hook. Drop the lead-list line from the arithmetic.

**Owner assets used.** The content-addressed daily-snapshot engine and GitHub self-commit pipeline (direct reuse); DNS-monitor, certificate-transparency, internet-outages collectors. Calibration engine not load-bearing here.

**30-day public probe.** Days 1-5: daily resolver for custom Tranco 1M + live CT hostnames into the existing store, self-committing daily summaries to a public repo. Days 5-10: free per-domain timeline page, per-IP attribution timeline from RIR/BGP/cloud-range diffs, PyPI client with free tier, subfinder provider PR. Days 10-30: one HN/Lobsters post, DNS-OARC and mailop lists, bug-bounty communities; $29-49/mo Stripe API key. Success: >2,000 unique domains looked up by strangers, >300 API-key signups, >=5 card payments, inbound asks for bulk/as-of/older history.

### 2.2 Open Security Master — survivorship-bias-free US listing/corporate-action master with a Form 144 → Form 4 insider ledger

**What it is.** Daily-snapshotted US security master (ticker/CIK/exchange/listing status, delistings with cause, symbol/name changes, splits, bankruptcies) reconciled from SEC EDGAR filings, shipped as versioned parquet + PyPI client + small API; second product on the same pipeline links every Form 144 (structured XML since April 2023) to its Form 4 execution with published out-of-sample hit-rates.

**Data sources and licence status.**
- SEC EDGAR (`company_tickers.json` diffs, Form 25, Form 15, 8-K Items 3.01/1.03/5.03, XBRL dei, Forms 3/4/5 and 144 XML): explicitly public, "may be copied or further distributed... without the SEC's permission"; "EDGAR" is a trademark and must not appear in the product name. https://www.sec.gov/privacy#dissemination
- FINRA Daily List and short volume: **the finalist was wrong** — FINRA Terms of Use restrict content to non-commercial personal/professional use, prohibit bulk copying, scraping and building databases from site content. Drop FINRA-sourced fields at launch or obtain written permission; the "three-source reconciliation" pitch becomes EDGAR-only. https://www.finra.org/terms-of-use
- OCC option-adjustment memos: terms unverifiable (403); treat as unlicensed until checked.
- CourtListener/RECAP bankruptcy dockets: owner already collects; fine for delisting-cause labels.

**Who pays and how they find it.** Systematic retail traders and 1-10 person funds who backtest (the Norgate/Sharadar buyer), RIAs, fintech developers, academics; event-driven traders and journalists for the insider ledger. Found via PyPI/GitHub (the open daily-diff repo is the funnel), quant communities, search ("delisted stocks list csv", "Form 144 data"). Channel is proven: sec-api.io Python SDK ~155K PyPI downloads/week (CI-inflated), edgartools ~190K/week. Card at $29-199/mo, $10K/yr platform licences.

**Verified comparable.** NONE FOUND at $1M/<=5 (the finalist said so honestly). Right-shape anchors verified: sec-api.io free / $49-55 personal / $199-239 business, Form 144 API in all paid tiers, solo-founder claim not independently verified (https://sec-api.io/pricing); Tiingo 1 employee per Tracxn, revenue undisclosed; Norgate Platinum $630/yr with delisted securities to 1990 (https://norgatedata.com/stockmarketpackages.php); Sharadar unfunded per its own About page only.

**Moat (after corrections).** Not the archive: Wayback already holds 96 monthly snapshots of `company_tickers.json` since 2017 (daily resolution is the only unique part, and it is a convenience file), and the primary events are permanent EDGAR filings with acceptance timestamps that anyone can reconstruct. What is defensible: cross-source reconciliation with a published error ledger, delisting *causes*, and the Form 144 → Form 4 linkage with calibrated hit-rates — nobody sells that, no GitHub repo versions the ticker file (0 results), and there is no free structured Form 144 dataset in the SEC Data Library (the free Insider Transactions Data Sets cover Forms 3/4/5 only). https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

**Honest kill risk.** Price ceiling: EODHD includes delisted tickers and a US symbol-rename-history API on all paid plans from $19.99/mo *bundled with prices* (https://eodhd.com/pricing); Massive (ex-Polygon) returns `delisted_utc` on every paid plan; Norgate at ~$50/mo-equivalent bundles delistings with prices. A price-less master alone is a <$30/mo product. sec-api.io already sells Form 144 + Form 4 + ticker map (with `isDelisted` flag, no dates, no PIT) to the same buyer and could ship a 144→4 join in weeks. Realistic $200-300K unless the bitemporal/adjusted-history layer follows.

**Revenue arithmetic.** 1,000 self-serve at $49 (~$590K) + 40 platform licences at $10K (~$400K); insider ledger adds ~1,200 at $49 if it lands. Treat $1M as a 3-year target riding on the bitemporal store.

**Owner assets used.** Insider-filings, Form 144, dividends, options-chain, bankruptcy-court and creditor-filing collectors (FINRA short-volume collector now excluded); content-addressed daily archive and self-commit pipeline; calibration/coverage engine for reconciliation error rates and insider hit-rates; the bitemporal store is the natural storage layer.

**30-day public probe.** Public repo committing a daily diff of `company_tickers.json` + Form 25/15 filings (new listings, delistings with reason, symbol changes) with parquet downloads, PyPI client, "delisted this week" page; free weekly "planned insider sales filed this week (Form 144) and which prior plans executed" page + CSV with linkage method and accuracy table; $9-19/mo Stripe for full history + API + alerts. One write-up ("every US delisting since 2015 reconciled from EDGAR — here are the disagreements with vendor lists") to quant communities and HN. Success: 300+ stars or 500+ PyPI installs/week, 500+ downloads, 10-20 paid, >=10 inbound requests for full history.

### 2.3 Open-AIS Vessel Ledger — Baltic shadow-fleet port-call archive + per-IMO identity ledger

**What it is.** Daily content-addressed archive computed from open-licence government AIS feeds: every tanker call at Russian Baltic terminals, Danish-strait exits, STS/loitering and AIS-gap events; per-IMO timeline of name/MMSI/flag/call-sign churn plus sanctions designations; berth-level dwell/throughput for US and Nordic terminals. Card at $99-999/mo with a public dated prediction ledger.

**Data sources and licence status.**
- Finnish Digitraffic AIS: CC BY 4.0, commercial use and redistribution explicit. Real-time only (no history parameter). **Coverage of Primorsk/Ust-Luga/Vysotsk from Finnish shore stations is asserted, not verified — this is the first thing the probe must check.** https://www.digitraffic.fi/en/terms-of-service/
- Norwegian Kystverket open AIS: NLOD, but *historical* data and a positions-by-MMSI-and-time API exist (Kystdatahuset), so Norwegian coverage is back-fillable; must use the open feed only (registered feed forbids third-party distribution). https://www.kystverket.no/en/sea-transport-and-ports/ais/access-to-ais-data/
- Danish DMA: "Historical AIS-data are free for down-load", no licence text, bulk host serves an expired TLS certificate; the "DMA deletes old files" claim is unverified. Danish-strait exits rest on an unstated licence. https://www.dma.dk/safety-at-sea/navigational-information/ais-data
- NOAA MarineCadastre: public domain, quarterly lag.
- Paris MoU inspection database: explicit prohibition on storing in a retrieval system or transmitting without written authorisation; EU database right makes "facts only" thin. **PSC-detention leg must use USCG PSIX (public domain) and drop/negotiate Paris and Tokyo MoU.** https://parismou.org/inspection-search/inspection-search
- IMO GISIS: behind login redirect, unverified. Equasis: excluded (commercial extraction forbidden).

**Who pays and how they find it.** Journalists/NGOs, indie oil/freight analysts, 1-10 person commodity/freight funds, small sanctions-compliance functions at brokers/charterers/P&I intermediaries, ship managers vetting counterparties. Found via free daily page + GitHub CSV, per-IMO pages indexed by Google ("IMO 9xxxxxx" is a known SEO channel — VesselFinder/BalticShipping rank on it), X/#OOTT, press citations, PyPI client. Note the licence split: journalists/NGOs/academics can already use Global Fishing Watch and OpenSanctions free tiers, so the *commercial* small buyer (broker, charterer, small fund) who cannot legally use those is the real customer.

**Verified comparable.** TankerTrackers.com: tiers verified — PremiumPlus $99/mo (closed to new signups 2023-01-01), Corporate Lite $1,000/mo, Corporate $5,000/mo, sold by "email us to begin signup" (inbound, no procurement). Revenue ~$4M is a RocketReach estimate only (profile 403 today); team 3-5 unverifiable; its moat is daily satellite imagery + shoreside photography, not open AIS. **It abandoned the exact $99 band this plan leans on.** https://tankertrackers.com/articles/subscription-service-tiers-explained

**Moat (after corrections).** Entity resolution over AIS static messages (name/MMSI/flag/call-sign churn) with calibrated event detection — a genuine solo-holdable correctness moat. The vintage moat shrinks to "Finnish-station view of Russian Gulf-of-Finland terminals from the day you start" (if Finnish coverage reaches them). The "open licence lets us sell at $99-499 where Spire/Kpler licensees cannot" claim is false: Datalastic (credit tiers, historical positions, port calls, inspections) and Pole Star On Demand ($250/screening by card) already sell in that band. https://datalastic.com/pricing/ ; https://www.polestarglobal.com/on-demand/

**Honest kill risk.** Coverage ends where open AIS ends (no Novorossiysk, Kozmino, Hormuz, Malacca); GFW gives port visits, loitering, encounters, AIS gaps and 40-registry vessel identity free under CC BY-NC (https://globalfishingwatch.org/our-apis/); OpenSanctions Maritime already holds 20,448 vessels with IMO/MMSI from 23 sources including Tokyo MoU (https://www.opensanctions.org/datasets/maritime/); KSE and IMF PortWatch cover headline numbers free. Open-source pull is weak (top repo 43 stars). If Digitraffic does not actually receive Primorsk/Ust-Luga/Vysotsk berth traffic, the core dataset does not exist.

**Revenue arithmetic.** ~280 subs at $299 avg; or 60 compliance seats at $999 + 120 at $199 + 80 at $99. Given TankerTrackers exited the $99 band, weight the arithmetic toward the $999 compliance tier and assume the $99 tier is a funnel, not revenue.

**Owner assets used.** AIS-vessels, sanctions, satellite-activity, energy-supply collectors; content-addressed daily-snapshot engine; calibration/coverage engine and dated prediction ledger for event detection and throughput forecasts.

**30-day public probe.** Day 0 (gate): confirm Digitraffic receives AIS from the three Russian terminals; if not, stop. Days 1-10: ingest Digitraffic + Kystverket + DMA; compute daily tanker arrivals/departures at Primorsk, Ust-Luga, Vysotsk, St Petersburg and Danish-strait exits; per-IMO timelines for the ~1,200 GUR-listed vessels from AIS static messages + PSIX. Days 10-30: free static page + repo with dated daily CSV, flag-hop leaderboard, per-IMO pages, RSS/newsletter, Stripe "vessel-level history + API $99/mo" link; one HN and one X/#OOTT post. Kill: <300 unique visitors/week after week 3, <25 newsletter signups, <1,000 organic clicks on IMO pages, zero inbound from a commercial (non-NGO) buyer.

### 2.4 Grid Asset Ledger — per-unit generator performance ledger on the queue spine, GPU site backtester as second product

**What it is.** Unit-level monthly ledger for every US utility-scale wind/solar/battery/thermal unit — capture price, curtailment MWh, outage hours, estimated energy+AS revenue — computed from ISO disclosures and EIA filings, crosswalked (ISO resource name → EIA plant/generator ID → queue ID) to the owner's daily interconnection-queue archive; second product: CUDA-batched "what would a battery/solar site have earned at any node" backtester with calibrated intervals.

**Data sources and licence status.**
- ERCOT: verified verbatim — public raw data "may be used, reproduced, and redistributed in compilations, charts, and analyses". Clean. https://www.ercot.com/help/terms
- EIA-923/860/860M, NRC, NREL: public domain. Clean.
- CAISO: every terms URL 404'd; unverified.
- PJM Legal & Privacy: access "does not confer any license or ownership interest... PJM hereby expressly reserves such rights" — no affirmative redistribution permission; Data Miner 2 terms not retrievable. **Verify before selling PJM-derived metrics.** https://www.pjm.com/about-pjm/legal.aspx
- ISO queue postings: captured by owner; LBNL Queued Up public.

**Who pays and how they find it.** Small developers and 1-10 person IPPs screening nodes/sites, energy consultants and land agents pricing PPAs, small funds/family offices underwriting storage, energy journalists/Substack analysts, academics. Found via free monthly asset league table, per-unit/per-node SEO pages, per-queue-project revision pages, gridstatus/PyPI ecosystem (small: 442 stars, 30 contributors), energy-analyst X/Substack. Card at $79-499/mo.

**Verified comparable.** NONE FOUND at $1M/<=5 (finalist's own statement, confirmed). PowerOutage.us is solo-run (https://www.yahoo.com/news/mans-side-project-became-source-132204804.html) but publishes no revenue and sells only via info@ enquiry — an inbound-quote business, not self-serve. Modo Energy $1.6M ARR at 36 employees (https://getlatka.com/companies/modo.energy). Grid Status ~$1.7M ARR at ~15 people, Professional/Enterprise are "Contact Sales" (https://getlatka.com/companies/gridstatus.io). **All three nearest firms converged on contact-sales pricing; nobody has demonstrated the $79-499 card-paid prosumer segment in grid data.**

**Moat (after corrections).** The "archive nobody can back-fill" leg is mostly false: CAISO archives its Curtailed and Non-Operational Generators reports back to June 2021 (https://www.caiso.com/market-operations/outages/curtailed-and-non-operational-generators); ERCOT 60-day SCED/DAM disclosures are downloadable by date via open-source gridstatus (`get_60_day_sced_disclosure`); GridTracker has captured every US queue daily since April 2024, so the owner's 2026-09-02 queue archive is the junior copy (https://www.gridtracker.io/). PUDL already integrates EIA 860/923/930, EPA CEMS and the EPA-EIA crosswalk free (CC-BY-4.0). What remains: the ISO-resource-name → EIA-ID → queue-ID crosswalk, unit-level capture-price/curtailment for wind/solar/thermal (a genuine gap — EIA-923 has no prices or curtailment), and GPU nodal backtest speed with honest intervals.

**Honest kill risk.** Modo gives BESS performance benchmarks away on its Free tier across ERCOT/CAISO/PJM (https://modoenergy.com/pricing), so the battery leg's revenue is ~zero. Grid Status hosts the identical raw datasets with 15 people and could ship per-asset scorecards in a quarter; the owner's public probe is a free spec for them. ERCOT's 60-day lag feels stale to traders; lenders want forecasts not backtests. Required comparable: none.

**Revenue arithmetic.** Finalist claimed ~$1.0M ledger + ~$1.0M backtester. After corrections: strike the BESS-benchmark line, treat the queue changelog as free funnel (interconnection.fyi is free and daily), and the realistic path is $300-500K year two, $1M year three, *only if* the wind/solar capture-price/curtailment per-unit product shows willingness to pay.

**Owner assets used.** Best fit in the set: the 6-of-7 ISO daily queue archive and self-commit pipeline (as spine, not product); the 1,636-row dated prediction ledger and calibration/coverage engine; power-grid, electricity-monitor, energy-supply, weather-alert, building-permit collectors; CUDA/PyTorch for batched dispatch.

**30-day public probe (redesigned per skeptic).** Test wind/solar per-unit willingness to pay, not battery leaderboards: (a) free monthly "ERCOT + CAISO wind/solar asset league table" (top/bottom 50 by capture price and curtailment, one page per unit, CSV) with a $79/mo "full history + all units" Stripe link; (b) precomputed "every ERCOT settlement point ranked by 2-hour battery revenue, last 12 months" page with a "run custom backtest" email queue (backtester demand, not benchmark demand); (c) per-queue-entry revision pages as free funnel. `pip install assetledger` with a free 90-day window; one Substack post on the crosswalk method; one X/HN post. Thresholds: >=2,000 uniques, >=100 per-unit page views from search, >=80 custom-backtest requests, >=60 full-history signups, >=5 card payments on the wind/solar tier, inbound "can you do PJM/MISO".

### 2.5 Manifest Ledger — US customs-manifest graph: imports-by-ticker nowcast + supplier exposure screening

**What it is.** Subscribe to the public CBP vessel-manifest feed (19 CFR 103.31(c)), entity-resolve shippers/consignees to tickers, brands and sanctioned/Entity-List/UFLPA parties, sell (a) weekly imports-by-ticker series to small funds/analysts and (b) per-importer exposure scores and "who imports from <listed entity>" pages to small importers, brokers, journalists, short-sellers.

**Data sources and licence status.**
- 19 CFR 103.31: manifests public; CBP sells compiled data at production cost, billed monthly in advance, no redistribution or commercial-use restriction; 103.31(d) confidential filers excluded. Clean, and the entire vendor industry rests on it. https://www.law.cornell.edu/cfr/text/19/103.31
- **The cited CBP PDF is not a fee sheet** — it is the "Sea AMS Data Processing Provider List" (Pub 0875-0419), a directory of eManifest software vendors. The subscription fee is unverified and must be obtained from CBP's Collections Section. This is the one finalist with a **recurring paid input**, requiring explicit cost-discipline approval; ImportYeti calls the FOIA route "super expensive".
- Overlay lists (OFAC, Entity List, UFLPA ~200 entities on dhs.gov): public domain.
- "Inference around confidential filers" is built on absence of data and de-anonymises parties who requested confidentiality — a reputational/legal grey zone.

**Who pays and how they find it.** Small importers, e-commerce sellers and journalists demonstrably pay by card for manifest search (ImportGenius 2008 pricing; ImportYeti's 100M+ TikTok views / 1.2M subreddit visits). The finance leg is *unproven*: no verified example of small funds paying by card for imports-by-ticker; that segment historically buys Panjiva via S&P sales. Found via free weekly imports-by-ticker charts timed to earnings, per-listed-entity supplier pages, GitHub CSVs, X during earnings week.

**Verified comparable.** ImportGenius: founders and 2006 founding verified, TechCrunch 2008 verifies $99/$250 monthly card tiers and organic spread from small importers to analysts and law firms (https://techcrunch.com/2008/05/28/importgenius-the-disruptive-shipping-database/). **The "$0 to $50K/month in two weeks" and "bootstrapped to millions" claims appear in none of the cited sources — revenue at the <=5-person stage: NONE FOUND.** ImportYeti: bootstrapped, no VC, cash-flow positive 2021-22, free core; headcount/pricing unverifiable.

**Moat (after corrections).** The "un-backfillable vintage" claim is largely false: bills of lading carry arrival dates and the full history is purchasable retroactively (ImportGenius sells 2006-present at $449/mo) or FOIA-able; "what was knowable before earnings date T" reconstructs from arrival date + fixed publication lag. Only the ticker/brand crosswalk state and confidential-filer inference as-of-date are unique — thin. The honest moat is packaging (finance-grade dated series + exposure scores by card at $50-1,000) plus resolution quality.

**Honest kill risk.** Panjiva (S&P) already advertises Xpressfeed to "feed data about publicly-traded companies into your quant models" (https://panjiva.com/) — product (a) is a price-band gap, not white space, closable by a pricing decision. ImportYeti could bolt free list-screening onto its free pages in weeks. Tens of millions of BoLs/yr is a real load for one person. Paid input of unknown size.

**Revenue arithmetic.** ~140 at $599; or 60 funds at $999 + 200 analysts at $149 + 300 importers/brokers at $49-199. Discount the fund line until the probe shows a fund paying by card.

**Owner assets used.** Sanctions, comtrade, insider-filings, gov-contracts collectors; content-addressed archive engine; calibration engine for nowcast intervals and a dated pre-earnings ledger.

**30-day public probe.** Before paying CBP: obtain the actual subscription fee and delivery latency from CBP (gate: latency must support weekly cadence). Days 1-10: resolve top 200 consumer/industrial tickers and top 5,000 importers from ImportYeti free pages / a single trial month; publish "Weekly US seaborne imports by ticker" for 50 names timed to earnings + "who imports from <listed entity>" pages + methodology note. Days 10-30: X during earnings week, Show HN; Stripe pre-order $149/mo; exposure-alert waitlist. Kill: <500 visitors/week, <20 pre-orders / <150 waitlist, no inbound from any fund, journalist or broker. Subscribe to the CBP feed only if the gate is met.

### 2.6 FedVintage — daily SAM.gov/FPDS snapshot archive with recompete calendar and award-to-ticker engine

**What it is.** Daily append-only snapshots of SAM.gov opportunities, FPDS awards/modifications, USAspending, FSRS, exclusions and GAO protests, preserving edits, cancellations and as-first-reported award values, with UEI/CAGE-to-parent-to-ticker resolution; sold as recompete calendar, award-event feed and metered API at $49-500/mo.

**Data sources and licence status.**
- SAM.gov Terms of Use: public data free of copyright; **automated scraping of sam.gov is prohibited** — the archive must be built from the public Opportunities API and Data Services extracts; D&B data (except D&B Open Data) may not be used for commercial/resale purposes — exclude legacy DUNS fields. https://sam.gov/content/about/terms-of-use
- FPDS-NG now redirects to sam.gov/contracting; same terms. USAspending: Treasury public domain.
- Public Get Opportunities API returns only the latest active version; "Get Opportunity History" exists on the Opportunity Management API, which requires a system account with a *government* email. https://open.gsa.gov/api/opportunities-api/

**Who pays and how they find it.** Small/mid federal contractors and capture consultants, indie defence/IT-services analysts, GovCon journalists. Self-serve card market confirmed: HigherGov Starter $500/yr and Standard $2,500/yr have "Subscribe" buttons (https://www.highergov.com/pricing/); Jorpex $49/$149/mo (https://jorpex.com/pricing/); GovTrove free. SEO on "expiring contracts NAICS", "recompete".

**Verified comparable.** GovTribe: $927.8K revenue (Apr 2021), 11 employees, $0 raised — bootstrapped, near the bar, not <=5 (https://getlatka.com/companies/govtribe). HigherGov: acquirer's blog says "grew without venture funding or a sales team"; revenue, headcount and price undisclosed; the "3,000+ contractors" figure is for the combined platform (https://www.procurementsciences.com/blog/procurement-sciences-acquires-highergov-one-platform-to-find-and-win-government-contracts). Strict comparable: partially met.

**Moat (after corrections).** Half false. The two skeptics disagree on SAM.gov: the government *does* retain full notice history, but only government-email accounts can pull it, so non-government parties cannot back-fill — the owner cannot either, so the archive starts empty. USAspending already publishes monthly Full and Delta archive files per agency to FY2007 plus transaction-level API (https://files.usaspending.gov/award_data_archive/). What is genuinely un-backfillable: pre-correction field values of FPDS records silently restated within a month, and contents of deleted records. Small, and nobody has shown anyone pays for it.

**Honest kill risk.** Commoditising fastest of the six: GovTribe sells "complete FPDS-derived award histories"; HigherGov already has Transaction History, Date Due filtering and real-time alerts at $500/yr; GitHub hosts free Recompete Radar, a DoD contract repo resolving subsidiaries to tickers via 3-pass entity resolution, and govconapi-mcp with 53 tools and 500K OCR'd solicitations (https://github.com/topics/sam-gov). The recompete calendar and award-to-ticker feed are commodity; only the restatement sliver is defensible, and it may be a curiosity. Lowest owner fit (3).

**Revenue arithmetic.** ~700 at $120 blended, or 1,000 seats at $79 — but HigherGov's $500/yr and Jorpex's $49/mo bound the price, and the differentiated sliver has no demonstrated buyer.

**Owner assets used.** Gov-contracts, lobbying, FOIA, regulatory-gazette collectors; content-addressed archive and self-commit pipeline; bankruptcy/creditor collectors for contractor-stress enrichment.

**30-day public probe (re-targeted per skeptic).** Test willingness to pay for restatement/deletion history specifically, not search or a calendar: Days 1-7 daily API-sourced snapshots of SAM.gov opportunities and FPDS deltas into a public repo. Days 7-20: free "What SAM.gov changed or deleted yesterday" page and a "restatement tracker" listing the 50 largest post-hoc FPDS corrections this month (the recompete calendar as funnel only). Days 20-30: Show HN, r/GovernmentContracting, GovCon LinkedIn; $49/mo "alerts + history export" Stripe. Success: >=1,000 daily uniques on the diff/restatement pages, >=200 signups, >=20 paying or >=20 inbound requests naming restatement/deletion history. Failure: <200 uniques/day or all inbound asks are for search/alerts.

---

## 3. Killed by skeptics

- **PDUFA Vintage** (moat-crowding, confidence 4): the entire moat — "competitors show current state only and cannot back-fill the announce-vs-actual ledger" — is false. BiopharmaWatch's live OpenAPI spec exposes an `as_of` parameter ("returns only decisions whose SOURCE DOCUMENT had published on or before this date") with revision chains (chainId/version/supersedes) on historical PDUFA and FDA-calendar endpoints, plus indication-specific PoA base rates, at API Basic $99/mo and API Elite Plus $189/mo, with 5+ years of decision outcomes. https://www.biopharmawatch.com/assets/openapi.json ; https://www.biopharmawatch.com/subscription . BPIQ APEX $55/mo includes Historical Catalysts + API + MCP. Remaining differentiator (calibration tooling) is a feature closable in a quarter. Comparable (BioPharmCatalyst, 3,000 paying subs) verified on subscribers but not on solo status or revenue.
- **Macro Release Vintage** (moat-crowding, confidence 3): the flagship "publishers overwrite" claim is false for the named sources — EIA STEO editions archived to 1983 with Excel (https://www.eia.gov/outlooks/steo/outlook.php), WPSR archived to 2011, Eurostat keeps 742 pages of first-print press-release PDFs; ALFRED/Philly Fed/ECB RTD/StatCan/OECD MEI cover the core free. The un-backfillable residue (EM/China overwrite-only series) is the thinnest buyer pool (<200 seats by the finalist's own estimate). FXMacroData (sole trader) already sells 26 years of point-in-time announcements for 17-18 currencies at $50/mo (https://fxmacrodata.com/). Seed community: macrotrace has 1 GitHub star. Comparable: NONE FOUND.

---

## 4. Dropped at merge

- **SanctionsVault** — OpenSanctions already keeps daily versioned exports since ~July 2021 retrievable by date and could ship as-of screening in a week; designation-timeline logic folded into the vessel and manifest ledgers.
- **MirrorGap (transshipment index)** — comparable NONE FOUND at <=5; 6-10 week lag; free think-tank substitutes; possible later second product on the Manifest Ledger.
- **DarkStats (satellite nowcasts)** — comparable NONE FOUND; noisy proxies, no track record, free X/Sentinel coverage; flaring/outage detection may feed other ledgers as a feature.
- **EUAmeter (voyage carbon cost)** — comparable NONE FOUND; incumbents bundle ETS calculators free; demand depends solely on regulation surviving.
- **Power-for-Load Atlas** — decisive number (substation headroom) sits in CEII-restricted data; crowded; no <=5 comparable (Data Center Map ~$521K); crosswalk subsumed by Grid Asset Ledger.
- **Docket-to-Queue** — comparable NONE FOUND; EnerKnol, GridTracker filing search and free FERC eSubscription leave too small a niche; doc-search UX not owner's strength.
- **Volatility Surface Vintage Archive** — is the data layer of the options surface lab already on the table; OPRA-derived terms unverified; ThetaData ships greeks/IV at $40-80/mo.
- **ETF Daily Holdings Vintage** — issuer ToS unverified and could kill it; ETF Global holds a 9-year archive; 200+ parsers are a permanent tax.
- **ComputeVintage (GPU/LLM price vintages)** — crowded with free trackers; weak permanence; only comparable (Keepa) is a different market.
- **Creditor Graph** — free dockets only for claims-agent cases, thin where mid-market suppliers would pay; incumbents serve deep pockets; collectors instead supply delisting-cause labels to Open Security Master.
- **Employer Events Vintage (WARN + postings)** — comparable NONE FOUND; free/cheap substitutes; career-page scraping ToS friction; enterprise competition.

---

## 5. Empty sub-fields

Internet measurement/security: cloud-storage exposure index (GrayHat Warfare pricing verified, no founder/revenue evidence, legal exposure); BGP/RPKI intelligence (bgp.tools one-man at GBP25/mo, revenue not found, raw layer free); outage/censorship index (IODA, Radar, Atlas free; NetBlocks grant-funded); NRD/typosquat feeds (crowded at $40-50, Zonefiles.io shut July 2025); CT subdomain search (Merklemap occupies it); breach-credential curation (legal/ethical exclusion); small-company attack-surface scanning (funded SaaS + free scanners); proxy/VPN IP classification (proxycheck.io/ipapi.is indie but no revenue evidence; ipinfo/IPQS dominate).

Geopolitics/OSINT: lobbying/FARA/FEC graph (LegiStorm bootstrapped but 26-45 staff; OpenSecrets/LobbyView free); shadow-fleet via satellite AIS (licence forbids derived resale — hence the open-feed-only vessel ledger); conflict/event datasets (ACLED/GDELT free; Factal VC; grant-cyclical buyers); political-risk indices (no card-paid small comparable; needs track record → discovery calls); disinformation tracking (pedigree-gated/grant buyers); FOIA/gazette monitoring standalone (free alerts, journalists rarely pay); migration/labour signals (free, no paying segment).

Maritime/supply chain: satellite/nightlight indices (no <=5 comparable, imagery licences); Comtrade/BACI anomaly monitor (OEC ~30 staff; Comtrade derived redistribution needs premium sub); chokepoint transit indices (IMF PortWatch free daily for 28 chokepoints); GFW-derived products (non-commercial only); global dark-activity/STS outside open coasts (satellite AIS licences); liner schedule reliability (Sea-Intelligence/Linerlytica own it).

Energy/grid: building-permit signals (Shovels, Construction Monitor, Ohm — no solo gap); REC/capacity prices (confidential in GATS/WREGIS); ISO forecast-vintage archive (Grid Status stores `publish_time` vintages); nodal LMP/congestion resale (Grid Status/Yes Energy; overlaps killed raw-data resale); FTR auction analytics (buyers are ISO members with procurement).

Finance derived data: PIT XBRL fundamentals (Sharadar, Calcbench, EODHD, FMP — cheap and crowded); earnings-date revision archive (Wall Street Horizon; folded as possible second product to the security master); 13F/institutional ownership (saturated); dividend history (free/cheap everywhere); delisting/queue-attrition prediction (kill list); **verified $1M+/<=5 comparable in finance data: only EODHD publishes revenue (~$3.5M) but at 11 employees — every finance comparable is flagged**.

Public records/corporate: company registries (Endole ~9 staff, OpenCorporates ~20, UK bulk free); Orange Book expiry data (DrugPatentWatch ~4 staff, revenue undisclosed, overlaps killed PDUFA lane); 510(k) data (free databases); building permits (VC-backed); non-bankruptcy dockets (CourtListener nonprofit; Docket Alarm undisclosed); job-posting-only signals (enterprise only).

Archives/vintages cross-cut: ToS/privacy-policy diffs (Open Terms Archive free + five indie tools); SaaS pricing-page history (SaaS Price Pulse, SaasTrack; partly Wayback-backfillable); weather/NWP vintages (Open-Meteo solo already sells previous-runs API); US/UK/Canada statistics vintages (ALFRED/RTDSM/StatCan/reviser/macrotrace); sanctions-list vintages (OpenSanctions); AI leaderboard vintages (fails permanence); retail price history (Keepa, 1 employee, EUR61.5M — the comparable pattern, not a slot); crypto venue listings (not permanent-need).

---

## 6. Pairings with the strict-scan survivors

Strict-scan survivors on the table: options surface lab, GPU backtester, retirement decumulation optimizer, bitemporal store, trade/tax engine.

**Named pairing: Open Security Master + bitemporal store + GPU backtester.** This is the only combination in the set where the shared engine is real, the licence is clean end-to-end (EDGAR-only), and the channel is verified (PyPI/GitHub → card, per sec-api.io, edgartools, Norgate, Sharadar). The bitemporal store *is* the storage layer the security master needs (the finalist says so); a survivorship-bias-free, corporate-action-correct universe is the input every backtester needs and is precisely what Norgate/Sharadar buyers pay for; the GPU backtester then has a differentiated input nobody else's backtester has (delisting causes, PIT reconciliation error ledger, 144→4 intent signals as a factor). The options surface lab also consumes the same master (OCC adjustment memos, splits, symbol changes — subject to OCC terms being cleared). Arithmetic: security master ~$1M at 3 years (finalist) + GPU backtester (strict scan, no independent revenue estimate) → $2M is arithmetic, not evidence. Neither half has a verified $1M/<=5 comparable; the pairing's strength is that one engineering effort (bitemporal EDGAR pipeline) feeds three products, not that any one is proven.

**Second pairing, weaker: Grid Asset Ledger + GPU backtester (as Site Backtester).** Same engine literally — the Site Backtester is the GPU backtester pointed at the LMP/AS archive. Finalist arithmetic claims ~$1M each. But after corrections the ledger's archive moat is struck, the BESS line is zero (Modo free), no comparable exists, and the three nearest incumbents all went contact-sales. This pairing has the best owner-asset fit and the worst evidence.

**DNS Wayback + bitemporal store.** As-of/diff/reverse queries over a content-addressed daily archive is a bitemporal query engine; the same store is what makes the IP-as-of projection cheap. This is an internal pairing (one store, two or three sellable projections after dropping lead lists), already counted inside the $1M target rather than a path to $2M. It shares nothing with the finance products.

**No pairing found for:** the decumulation optimizer (no data overlap with any finalist) and the trade/tax engine (would consume the security master's corporate-action history, but that is a dependency, not a shared product).

Is there a two-product pairing on shared engine or data that *reaches* $2M+ on evidence? **None.** The Security Master + bitemporal + GPU backtester stack is the one worth building toward because one pipeline yields three card-paid products in a proven channel, but its $2M is a three-year arithmetic target with zero verified comparables in the exact niche.

---

## 7. Cross-cutting patterns

1. **"Archive nobody can back-fill" was the single most refuted claim in the scan.** It failed wholly or mostly for CAISO, ERCOT, Danish/Norwegian AIS, EIA, Eurostat, SAM.gov, USAspending, bills of lading, `company_tickers.json`, and PDUFA dates. It survived only where the owner *performs the measurement* (DNS resolution from his own resolvers, Finnish real-time AIS reception, FPDS pre-correction values, daily-resolution ticker diffs, SAM.gov notice history for non-government accounts). Rule for future scans: a vintage archive is a moat only if the primary record is transient *and* the owner is the sensor. If the primary record is a permanent filing or a permanently hosted PDF, the archive is a convenience, not a moat.

2. **Licence errors clustered in the aggregator layer, not the raw layer.** Raw government sources (SEC, EIA, ERCOT, Digitraffic, CBP, NOAA, FDA) were consistently clean. The finalists went wrong on the convenience aggregators: FINRA Daily List (non-commercial ToS), CZDS (no marketing to non-customers), Tranco (embeds CC BY-NC Radar data), ARIN/RIPE bulk WHOIS, Paris MoU (retrieval-system prohibition), Eurostat (excludes non-EU country data from commercial reuse). Every future candidate should be licence-checked at the aggregator it actually plans to fetch from, not at the originating authority.

3. **The strict "$1M / <=5 / no outbound / cited URL" comparable rule produced NONE FOUND in every exact niche.** What exists are adjacent-lane shape matches (ipinfo, BuiltWith, sec-api.io, Tiingo, Trading Economics at 3 employees, GovTribe at 11, TankerTrackers at an unaudited estimate). Revenue figures for indie data companies are almost universally third-party estimates (ZoomInfo/RocketReach/Latka) that could not be re-fetched. The relaxation to $1M did not unlock verified comparables; it lowered the bar to a level at which the evidence is simply not published.

4. **Two price convergences bracket the target band.** Cheap self-serve incumbents converge at $20-55/mo (BPIQ, BiopharmaWatch, EODHD, ViewDNS $29, CompleteDNS $30, Merklemap EUR49, Jorpex $49, FXMacroData $50); enterprise incumbents converge at contact-sales (Grid Status, Modo, PowerOutage.us, GovTribe, TankerTrackers at $1k-5k). The $50-1,000/mo card-paid mid-band the brief targets is demonstrated only in DNS/IP data (SecurityTrails $500-1,500, BuiltWith $295-995, ViewDNS $549) and gov-con ($500-2,500/yr). In grid, maritime and biotech data, nobody has shown that band exists.

5. **Free-tier-as-funnel from better-funded incumbents recreates the "developer engines die" pattern the strict scan found.** Modo (BESS benchmarks free), Global Fishing Watch (CC BY-NC free), OpenSanctions (free NC + PAYG), ImportYeti (free core), GovTrove (free), interconnection.fyi (free), dns.coffee (free), CatalystAlert (free, then dead — the low end does not sustain). A solo entrant's public probe is also a free spec for the incumbent hosting the same raw data (Grid Status, Merklemap, ImportYeti, sec-api.io were each named as "could ship in a quarter").

6. **Owner-asset reuse is highest exactly where evidence is weakest.** Grid Asset Ledger (fit 5, uses the queue archive, prediction ledger, calibration engine, CUDA) has no comparable and a struck moat; DNS Wayback (fit 5, reuses the snapshot engine) has the best comparable but competes with the owner's archive at day zero against decade-deep incumbents. The calibration/coverage engine — the owner's most distinctive asset — is load-bearing only in the vessel ledger (event detection), the security master (reconciliation error rates, insider hit-rates) and the grid backtester (intervals). It is decorative in DNS, manifests and FedVintage.

7. **What the relaxation actually bought.** One lane opened cleanly: derived data where the owner is the sensor or the reconciler (DNS resolution; EDGAR reconciliation; AIS entity resolution). One lane opened with a paid input and cost-discipline approval required (CBP manifests). The inbound-selling relaxation matters only in the DNS and gov-con lanes, where card-paid mid-band buyers are demonstrated; elsewhere it changes nothing because the buyer band is unproven. The $1M target matters only if the second product shares the pipeline — which is why the recommended build direction is the EDGAR bitemporal pipeline (security master → backtester input → options-lab input), not any single finalist.

**Recommended next action.** Run the two cheapest probes in parallel — Open Security Master (EDGAR-only daily diff repo + Form 144→4 page; zero paid inputs, zero unverified licences) and DNS Wayback (resolver + per-domain/per-IP pages + subfinder PR) — for 30 days, and gate the Grid Asset Ledger probe on the Day-0 ERCOT wind/solar per-unit page only. Do not pay CBP, and do not run the vessel probe until Digitraffic coverage of the three Russian terminals is confirmed.
