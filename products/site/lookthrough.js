/* Look-through, computed in the browser.
 *
 * WHAT THIS IS
 *   Given a pasted list of holdings and the precomputed tables in
 *   data/lookthrough.json, work out what the book really holds once index
 *   funds are unpacked: for every company,
 *
 *       direct weight  +  sum over funds of (fund weight x its weight in the index)
 *
 *   No network, no backend, no price history. The look-through needs index
 *   membership and index weights, both of which are precomputed, so it runs
 *   on a static page. The picking-cost and factor blocks genuinely need
 *   price history and stay on the server.
 *
 * WHY IT IS SAFE TO HAVE A SECOND IMPLEMENTATION
 *   Normally it is not. Two implementations of one calculation is how two
 *   answers to one question appear, and the project's own failure log is
 *   mostly variations on that. The guard here is specific and mechanical:
 *
 *     1. This file holds NO data. Every table, phrase list and regex comes
 *        from lookthrough.json, which scripts/export_lookthrough_data.py
 *        serialises out of the Python modules the server uses. There is one
 *        source of truth for the tables and it is Python's.
 *     2. tests/test_lookthrough_static_parity.py runs the functions below,
 *        under node, over every scheme name in the live AMFI master and over
 *        whole golden books, and requires EXACT agreement with Python —
 *        phrase for phrase, and weights to 1e-12. Thousands of real names.
 *
 *   So this file is a port of logic whose agreement is proven on every
 *   build, not a reimplementation that is believed to agree.
 *
 * WHAT IT REFUSES
 *   Share counts and mixed currencies need a price per unit, and a fund has
 *   no price here, so those are refused with a reason rather than guessed.
 *   An index whose constituent list is not in the data file is refused by
 *   name. Under-claiming is recoverable; unpacking a position into 150 wrong
 *   companies and reporting success is not.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.LookThrough = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------------
  // Text normalisation — ports agent/lookthrough/funds.py
  // ---------------------------------------------------------------------

  var DASHES = ["‐", "‑", "‒", "–", "—", "―", "−"];

  /* Python's _nfkc. The dash and quote folding is not cosmetic: scheme names
     arrive with en-dashes from PDFs and curly apostrophes from spreadsheets,
     and a plan suffix written "– Direct Plan" must strip the same as
     "- Direct Plan" or the two forms of one fund resolve differently. */
  function nfkc(s) {
    s = (s == null ? "" : String(s)).normalize("NFKC");
    for (var i = 0; i < DASHES.length; i++) s = s.split(DASHES[i]).join("-");
    return s.split("‘").join("'").split("’").join("'").split(" ").join(" ");
  }

  function fold(s) {
    return nfkc(s).trim().toLowerCase().replace(/\s+/g, " ");
  }

  function stripExchangeSuffix(s) {
    var low = s.toLowerCase();
    var sufs = [".ns", ".bo", ".nse", ".bse", "-eq", ".eq"];
    for (var i = 0; i < sufs.length; i++) {
      if (low.endsWith(sufs[i])) return s.slice(0, s.length - sufs[i].length);
    }
    return s;
  }

  function splitGluedDigits(s) {
    return s.replace(/(?<=[a-z])(?=\d)/g, " ");
  }

  // ---------------------------------------------------------------------
  // The engine, built around one exported data blob
  // ---------------------------------------------------------------------

  function Engine(data) {
    if (!data || !data.resolver || !data.indices) {
      throw new Error("lookthrough data blob is missing resolver or indices");
    }
    this.d = data;
    var p = data.resolver.patterns;
    // Compiled from the Python pattern source, so the two engines read one
    // pattern. The parity corpus is what proves they agree on it.
    this.rePlan = new RegExp(p.plan_suffix, "i");
    this.reQty = new RegExp(p.qty_suffix, "i");
    this.reCurrency = new RegExp(p.currency_lead, "gi");
    this.reTicker = new RegExp(p.ticker);
    this.reUnitTail = /(?:units?|unit|shares?|shs?|qty|nos?|%|pct|percent|rs\.?|inr|₹|\$|usd)\s*$/i;
    this.reIndianGrouping = new RegExp(data.resolver.indian_grouping_rx);
    this.reScaleWord = new RegExp(data.resolver.scale_word_rx, "i");
    this.reNumericSize = new RegExp(data.resolver.numeric_size_rx, "i");
    this.sizeWords = Object.create(null);
    var sw = data.resolver.size_words || [];
    for (var n = 0; n < sw.length; n++) this.sizeWords[sw[n]] = true;
    // Longest house first, so "HDFC Mutual Fund" strips before "HDFC".
    this.houses = (data.resolver.house_prefixes || []).slice().sort(function (a, b) {
      return b.length - a.length;
    });
    // The line classifier, in the order Python tries it: first match wins, so
    // the specific index must come before the general one.
    var compile = function (rows) {
      return (rows || []).map(function (p) {
        return { rx: new RegExp(p.rx, "i"), index: p.index, ticker: p.ticker, why: p.why };
      });
    };
    this.indexPatterns = compile(data.resolver.index_patterns);
    this.opaquePatterns = compile(data.resolver.opaque_patterns);
    this.opaqueFallbackWhy = data.resolver.opaque_fallback_why || "";
  }

  /* Split one line into (name, size), mirroring Python's _split_size. The
     size is a run at EITHER end holding at most one number plus adjacent unit
     words; stopping at the first number keeps "Nifty 50" and "Midcap 150"
     intact instead of reading the 50 as a quantity.

     A size token must be a NUMBER, not any token containing a digit. That
     distinction is F-19(d): Indian ETF tickers are full of digits —
     SETFNIF50, MID150BEES, NIFTY1, HDFCNIF100 — and a "contains a digit" test
     ate the ticker itself, leaving an empty name that matched no fund pattern.
     Twenty of twenty-six known ETFs were classified as companies. */
  Engine.prototype.splitSize = function (text) {
    var toks = text.split(/\s+/).filter(Boolean);
    var self = this;
    var isUnit = function (t) { return self.sizeWords[t.replace(/^[.,:]+|[.,:]+$/g, "").toUpperCase()] === true; };
    var isNum = function (t) { return self.reNumericSize.test(t.replace(/^[.,:;]+|[.,:;]+$/g, "")); };

    var j = toks.length, seen = false;
    while (j > 0) {
      var t = toks[j - 1];
      if (isUnit(t)) { j--; continue; }
      if (isNum(t) && !seen) { j--; seen = true; continue; }
      break;
    }
    var i = 0; seen = false;
    while (i < j) {
      var u = toks[i];
      if (isUnit(u)) { i++; continue; }
      if (isNum(u) && !seen) { i++; seen = true; continue; }
      break;
    }
    return [toks.slice(i, j).join(" "), toks.slice(0, i).concat(toks.slice(j)).join(" ")];
  };

  /* Classify one line: a stock, a fund we can unpack, or a fund we can name
     but cannot see into. Mirrors Python's resolve_line step for step.

     Every table here is EXPORTED, not hand-written. The first version of this
     function used a hand-rolled /\b(fund|etf|bees|gold|...)\b/ and read
     GOLDBEES as a company, because \bbees\b does not match inside "GOLDBEES".
     A gold ETF was being counted among the reader's equity holdings. */
  Engine.prototype.classifyLine = function (text) {
    var cleaned = text.split(/\s+/).filter(Boolean).join(" ").replace(/^[\s,;\t•*-]+|[\s,;\t•*-]+$/g, "");
    var parts = this.splitSize(cleaned);
    var name = parts[0];
    var probe = name || cleaned;
    var r = this.d.resolver;
    var i;

    /* 0. The ticker tables, which are authoritative: each entry records the
          name the ISSUER lists the ETF under, read live and quoted verbatim,
          with the index derived from that name by the same reduction a pasted
          scheme name goes through. Consulted before the patterns because the
          pattern list only ever named 8 tickers while this table knows 25. */
    var bare = probe.trim().toUpperCase();
    if (bare && bare.indexOf(" ") < 0) {
      /* Non-equity ETFs first: "it holds metal" is a better answer for the
         reader than "a fund we cannot see into", and it is the one case where
         we can say something definite about the contents. Listed rather than
         inferred from spelling — \bgold\b misses GOLDBEES, and \bgold
         without the boundary matches GOLDIAM, a real listed company. */
      if (Object.prototype.hasOwnProperty.call(r.non_equity_tickers || {}, bare)) {
        return { kind: "opaque_fund", label: probe, index: null, why: r.non_equity_tickers[bare] };
      }
      if (Object.prototype.hasOwnProperty.call(r.opaque_tickers, bare)) {
        return {
          kind: "opaque_fund",
          label: probe,
          index: null,
          why:
            bare + ' is a traded ETF, but the name it is listed under is "' +
            r.opaque_tickers[bare] + '", which names no index. Its issuer code ' +
            "hints at one; a hint is not a holdings list, so it was left " +
            "un-unpacked rather than mapped to a guess.",
        };
      }
      var listed = r.ticker_listed_names[bare];
      if (listed) {
        var tIdx = r.alias_to_index[this.indexPhrase(listed)];
        if (tIdx) {
          return {
            kind: "index_fund",
            label: probe,
            index: tIdx,
            why: bare + ' is listed as "' + listed + '", which tracks ' + tIdx,
          };
        }
      }
    }

    /* 1. The specific patterns. A NAME-based match is additionally vetoed by
          the reduced phrase: the pattern asks "does this name contain my
          index's name", the phrase asks "and nothing else that would make it a
          different index". F-19(b): \bnifty\s*50\b matches "Nifty 50 Equal
          Weight Index Fund", which holds the same fifty companies at 2% each
          rather than at the free-float weights this page would apply.

          A TICKER pattern is exempt — "SETFNIFBK" is an exact identity whose
          reduction is meaningless, and vetoing it would lose real coverage. */
    for (i = 0; i < this.indexPatterns.length; i++) {
      var ip = this.indexPatterns[i];
      if (!ip.rx.test(probe)) continue;
      if (!ip.ticker) {
        if (r.alias_to_index[this.indexPhrase(probe)] !== ip.index) continue;
      }
      return { kind: "index_fund", label: probe, index: ip.index, why: ip.why || "" };
    }

    /* 2. The phrase path, for a name that identifies its index without
          matching a pattern — "UTI Nifty Index Fund" reduces to "nifty", an
          alias of NIFTY 50. Only an EXACT alias hit counts. */
    var phrase = this.indexPhrase(probe);
    var phraseIndex = r.alias_to_index[phrase];
    if (phraseIndex) {
      var why = "";
      if (phrase === "nifty" || phrase === "cnx nifty" || phrase === "s p cnx nifty") {
        why =
          'the name carries no index number, and an unqualified "Nifty" is read ' +
          "here as the NIFTY 50 — which is the Indian convention, not a certainty";
      }
      return { kind: "index_fund", label: probe, index: phraseIndex, why: why };
    }

    /* 3. Things we can name but cannot see into. Breadth is safe HERE and
          nowhere above: a row in this table names no index, so a broad match
          says "I cannot see into this", which is the correct default for
          something we failed to identify. */
    for (i = 0; i < this.opaquePatterns.length; i++) {
      var op = this.opaquePatterns[i];
      if (op.rx.test(probe)) {
        return { kind: "opaque_fund", label: probe, index: null, why: op.why || this.opaqueFallbackWhy };
      }
    }
    return { kind: "direct", label: probe, index: null, why: "" };
  };

  Engine.prototype.stripPlanSuffix = function (s) {
    var prev = null;
    while (prev !== s) {
      prev = s;
      s = s.replace(this.rePlan, "").trim();
    }
    return s;
  };

  /* Python's _strip_qty_suffix. The "only accept the cut if a unit word was
     there" rule is load-bearing in both: a bare trailing number may be part
     of the index name, and stripping it turns "NIFTY 50" into "nifty". */
  Engine.prototype.stripQtySuffix = function (s) {
    s = s.replace(this.reCurrency, " ");
    var prev = null;
    while (prev !== s) {
      prev = s;
      var cut = s.replace(this.reQty, "").trim();
      if (cut !== s && this.reUnitTail.test(s)) s = cut;
    }
    return s.trim();
  };

  Engine.prototype.stripHouse = function (s) {
    for (var i = 0; i < this.houses.length; i++) {
      var house = this.houses[i];
      if (s === house) return [s, null];
      var seps = [" ", " - ", "- ", "-", " | "];
      for (var j = 0; j < seps.length; j++) {
        var lead = house + seps[j];
        if (s.startsWith(lead)) return [s.slice(lead.length).trim(), house];
      }
    }
    return [s, null];
  };

  /* Reduce a scheme name or ticker to the index it names, and nothing else.
     Lossy and deliberately so — only an EXACT hit against the alias table
     means anything, and the phrase is otherwise just a string to quote back
     to the reader. */
  Engine.prototype.indexPhrase = function (text) {
    var s = fold(text);
    if (!s) return "";
    s = stripExchangeSuffix(s);
    s = this.stripQtySuffix(s);
    s = this.stripPlanSuffix(s);
    s = this.stripHouse(s)[0];
    s = this.stripPlanSuffix(s);

    var phrases = this.d.resolver.boilerplate_phrases || [];
    for (var i = 0; i < phrases.length; i++) s = s.split(phrases[i]).join(" ");

    s = s.replace(/[^a-z0-9]+/g, " ");
    s = splitGluedDigits(s);

    var stop = this.d.resolver.boilerplate_words || [];
    var stopSet = Object.create(null);
    for (var k = 0; k < stop.length; k++) stopSet[stop[k]] = true;

    var words = s.split(" ").filter(function (w) {
      return w && !stopSet[w];
    });
    s = words.join(" ");
    if (s.indexOf("bank nifty") === 0) s = "nifty bank" + s.slice("bank nifty".length);
    return s.trim();
  };

  /* "index", "active" or "unknown". `index` does NOT mean we can unpack it:
     a gold ETF and a G-Sec index fund are both `index` here and neither
     holds a company share. The unpack decision is the alias table, below. */
  Engine.prototype.classify = function (schemeName, category) {
    var r = this.d.resolver;
    var name = fold(schemeName);
    var cat = fold(category || "");
    var i;

    if (cat) {
      for (i = 0; i < r.passive_category_markers.length; i++) {
        if (cat.indexOf(r.passive_category_markers[i]) >= 0) return "index";
      }
    }
    // Name before category: AMFI files an index-tracking fund-of-funds under
    // a category that says nothing about who picks the holdings. Its name does.
    var padded = " " + name + " ";
    for (i = 0; i < r.passive_markers.length; i++) {
      if (padded.indexOf(r.passive_markers[i]) >= 0) return "index";
    }
    if (cat && cat.indexOf("(") >= 0) return "active";

    var phrase = this.indexPhrase(schemeName);
    if (phrase) {
      for (i = 0; i < r.index_family_prefixes.length; i++) {
        var fam = r.index_family_prefixes[i];
        if (phrase === fam || phrase.indexOf(fam + " ") === 0) return "index";
      }
    }
    for (i = 0; i < r.active_markers.length; i++) {
      if (name.indexOf(r.active_markers[i]) >= 0) return "active";
    }
    return "unknown";
  };

  Engine.prototype.isNonEquityPhrase = function (phrase) {
    var ps = this.d.resolver.non_equity_phrases || [];
    for (var i = 0; i < ps.length; i++) if (phrase.indexOf(ps[i]) >= 0) return true;
    return false;
  };

  // ---------------------------------------------------------------------
  // Fund resolution
  // ---------------------------------------------------------------------

  /* Resolve one pasted fund, ALWAYS with a reason.
   *
   * Narrower than the server on purpose: the server also consults AMFI's
   * scheme master by ISIN and by exact name, which gives it AMFI's own
   * category. That file is far too large to ship to a browser, so this path
   * resolves on the typed text alone. The consequence is one-directional —
   * a fund whose name does not itself name its index is REFUSED here where
   * the server might have unpacked it. A refusal the reader can see is a
   * recoverable outcome; a wrong unpack is not.
   */
  Engine.prototype.lookUpFund = function (text) {
    var typed = nfkc(text).trim();
    if (!typed) {
      return {
        typed: typed,
        index: null,
        unpacked: false,
        reason: "an empty line was passed where a fund name was expected.",
        classification: "unknown",
        phrase: "",
      };
    }

    var r = this.d.resolver;
    var bare = stripExchangeSuffix(typed).trim().toUpperCase();

    // A bare exchange ticker resolves via the name it is LISTED under, never
    // via a hand-written ticker -> index table. The listed name is a quote
    // from the issuer; a hand-written mapping is an unverifiable claim.
    if (this.reTicker.test(bare) && typed.indexOf(" ") < 0) {
      if (Object.prototype.hasOwnProperty.call(r.opaque_tickers, bare)) {
        return {
          typed: typed,
          index: null,
          unpacked: false,
          reason:
            bare +
            ' is a traded ETF, but the name it is listed under is "' +
            r.opaque_tickers[bare] +
            '", which names no index. Its issuer code hints at one; a hint is ' +
            "not a holdings list, so it was left un-unpacked rather than " +
            "mapped to a guess.",
          classification: "index",
          phrase: "",
        };
      }
      var listed = r.ticker_listed_names[bare];
      if (listed) {
        var tphrase = this.indexPhrase(listed);
        var tindex = r.alias_to_index[tphrase];
        if (tindex && this.d.indices[tindex]) {
          return {
            typed: typed,
            index: tindex,
            unpacked: true,
            reason: bare + ' is listed as "' + listed + '", which tracks ' + tindex + ".",
            classification: "index",
            phrase: tphrase,
            listed_name: listed,
          };
        }
      }
    }

    var phrase = this.indexPhrase(typed);
    var indexId = r.alias_to_index[phrase];
    var cls = this.classify(typed, null);

    if (indexId && this.d.indices[indexId]) {
      var note = "";
      if (phrase === "nifty" || phrase === "cnx nifty" || phrase === "s p cnx nifty") {
        note = ' (that text carries no index number, and an unqualified "Nifty" is read here as the Nifty 50)';
      } else if (phrase !== (r.supported_indices[indexId] || [])[0]) {
        note = ' (matched on the alias "' + phrase + '")';
      }
      return {
        typed: typed,
        index: indexId,
        unpacked: true,
        reason: '"' + typed + '" tracks ' + indexId + note + ".",
        classification: "index",
        phrase: phrase,
      };
    }

    return {
      typed: typed,
      index: null,
      unpacked: false,
      reason: this.whyNot(typed, phrase, cls),
      classification: cls,
      phrase: phrase,
    };
  };

  /* The sentence a reader sees when their fund was left alone. States what
     they typed, what we understood it to be, and what is missing. Never
     recommends anything — this product is an audit, not advice. */
  Engine.prototype.whyNot = function (typed, phrase, cls) {
    var known = Object.keys(this.d.indices);
    if (!phrase) {
      return (
        'I could not read "' +
        typed +
        '" as a fund or an index at all — nothing was left of it after ' +
        "removing the fund house and the words index/fund/ETF. It was left " +
        "un-unpacked."
      );
    }
    if (this.isNonEquityPhrase(phrase) && cls === "index") {
      return (
        '"' +
        typed +
        '" tracks "' +
        phrase +
        '", which is not an equity index, so it holds no company shares to ' +
        "add to your stock positions. It was left un-unpacked."
      );
    }
    if (cls === "index") {
      return (
        '"' +
        typed +
        '" is a passive product tracking "' +
        phrase +
        '". I hold published constituent lists for only ' +
        known.length +
        " indices (" +
        known.join(", ") +
        "), and that is not one of them, so it was left un-unpacked rather " +
        "than mapped to a nearby index."
      );
    }
    if (cls === "active") {
      return (
        '"' +
        typed +
        '" is an actively managed fund — a manager picks its holdings. SEBI ' +
        "requires it to publish them monthly, but they are not knowable from " +
        "its name, and guessing them from its category would be inventing " +
        "data. It was left un-unpacked."
      );
    }
    return (
      'I could not tell what "' +
      typed +
      '" is. It reduced to "' +
      phrase +
      '", which matches no index I hold a constituent list for and carries ' +
      "none of the words that mark a fund (index, ETF, flexi cap, large " +
      "cap, ...). It was left un-unpacked."
    );
  };

  // ---------------------------------------------------------------------
  // Parsing the paste
  // ---------------------------------------------------------------------

  /* One line -> {name, number, kind}. Deliberately narrow.
   *
   * Indian digit grouping and scale words are REFUSED, not repaired: guessing
   * whether "1,20,000" means 120000 or 1.2 is exactly the guess that produces
   * a confidently wrong headline, and the server refuses them for the same
   * reason. Both patterns come from the data file so there is one source of
   * truth for what counts as unreadable. */
  /* Split with the SAME routine the classifier uses, rather than a regex of
     its own. An earlier version matched a trailing number only, so a paste of
     "50% HDFCBANK" — size first, which is how some broker exports and most
     hand-typed lists read — came back as "has no number on it". splitSize
     scans both ends, as Python's _split_size does, and sharing it means the
     name the size is stripped from is the same name the classifier sees. */
  Engine.prototype.parseLine = function (raw) {
    var line = nfkc(raw).trim();
    if (!line) return null;

    var cleaned = line.split(/\s+/).filter(Boolean).join(" ").replace(/^[\s,;\t•*-]+|[\s,;\t•*-]+$/g, "");
    var parts = this.splitSize(cleaned);
    var name = parts[0].replace(/[,:|]+$/, "").trim();
    var size = parts[1].trim();

    if (!size) return { error: "has no number on it" };
    if (!name) return { error: "has a number but no holding on it" };

    var m = size.match(/^(?:(rs\.?|inr|₹|\$|usd)\s*)?(-?[\d][\d,]*(?:\.\d+)?)\s*(%|pct|percent|units?|unit|shares?|shs?|qty|nos?)?\s*$/i);
    if (!m) return { error: "has a size this page cannot read: " + size };

    var cur = m[1] ? m[1].toLowerCase() : null;
    var num = parseFloat(m[2].replace(/,/g, ""));
    var unit = m[3] ? m[3].toLowerCase() : null;
    if (!isFinite(num)) return { error: "has an unreadable number" };
    if (num < 0) return { error: "has a negative size, which this page cannot interpret" };

    var kind = "bare";
    if (unit && /^(%|pct|percent)$/.test(unit)) kind = "percent";
    else if (unit) kind = "shares";
    else if (cur) kind = "amount";

    return { name: name, number: num, kind: kind, currency: cur, raw: line };
  };

  /* Decide one unit mode for the whole paste. Mixed modes are refused: a book
     half in percent and half in share counts has no single denominator, and
     inventing one silently rescales every number. */
  function unitMode(entries) {
    var kinds = {};
    entries.forEach(function (e) {
      kinds[e.kind] = (kinds[e.kind] || 0) + 1;
    });
    var present = Object.keys(kinds);
    if (present.length === 1) return present[0];
    // "bare" alongside percent is the common paste ("HDFCBANK 12%\nINFY 8").
    var nonBare = present.filter(function (k) {
      return k !== "bare";
    });
    if (nonBare.length === 1) return nonBare[0];
    return "mixed";
  }

  // ---------------------------------------------------------------------
  // The product
  // ---------------------------------------------------------------------

  function refuse(reason, extra) {
    var out = { computed: false, not_computed_reason: reason };
    if (extra) for (var k in extra) out[k] = extra[k];
    return out;
  }

  /* Add up direct holdings and fund holdings into one real weight per company.
   *
   * Post-look-through weights sum to 1.0 and the check is ENFORCED below, not
   * assumed: every number on the page is a share of that total, so a vector
   * summing to 0.97 understates every company by 3% while looking entirely
   * plausible. */
  Engine.prototype.lookThrough = function (text) {
    var self = this;
    var rawLines = String(text || "")
      .split("\n")
      .filter(function (l) {
        return l.trim();
      });
    var nListed = rawLines.length;
    if (!nListed) return refuse("nothing was pasted.");

    /* A number we would read wrong by orders of magnitude refuses the WHOLE
       paste, not just its own line — the same as the server. Skipping the bad
       line and computing the rest looks helpful and is worse: every remaining
       weight is a share of a total that silently lost a position, so all of
       them are wrong and none of them is marked. */
    var badNumbers = [];
    rawLines.forEach(function (raw, i) {
      if (self.reIndianGrouping.test(raw)) {
        badNumbers.push(
          'line ' + (i + 1) + ' "' + raw.trim() + '" uses Indian digit grouping, which reads as ' +
            'the digits before the first comma (so 1,20,000 becomes 1) — re-paste the number without commas'
        );
      }
      var scale = raw.match(self.reScaleWord);
      if (scale) {
        badNumbers.push(
          'line ' + (i + 1) + ' "' + raw.trim() + '" is sized in ' + scale[0] +
            ', which is dropped (so "2 ' + scale[0] + '" becomes 2) — write the number out in full, or use percentages'
        );
      }
    });
    if (badNumbers.length) {
      return refuse(
        "we will not put a number on this paste because one of its numbers would " +
          "be read wrong by five or six orders of magnitude: " + badNumbers.join("; ")
      );
    }

    var entries = [];
    var unreadable = [];
    rawLines.forEach(function (raw, i) {
      var p = self.parseLine(raw);
      if (!p) return;
      if (p.error) unreadable.push("line " + (i + 1) + ' "' + raw.trim() + '" ' + p.error);
      else entries.push(p);
    });
    if (!entries.length) {
      return refuse(
        "no readable positions in the pasted text" +
          (unreadable.length ? " — " + unreadable.join("; ") : "."),
        { unreadable: unreadable }
      );
    }

    var mode = unitMode(entries);
    if (mode === "mixed") {
      return refuse(
        "this paste mixes percentages, amounts and share counts, which have no " +
          "single denominator. Re-paste using one of them.",
        { unreadable: unreadable }
      );
    }
    if (mode === "shares") {
      return refuse(
        "this paste is sized in share counts. Turning those into weights needs " +
          "a price per unit for every line, and a fund has no price here, so " +
          "this page does not guess one. Re-paste as percentages.",
        { unreadable: unreadable }
      );
    }
    if (mode === "amount") {
      var curs = {};
      entries.forEach(function (e) {
        if (e.currency) curs[e.currency] = true;
      });
      if (Object.keys(curs).length > 1) {
        return refuse(
          "this paste mixes currencies, which needs an exchange rate this page " +
            "does not fetch. Re-paste as percentages.",
          { unreadable: unreadable }
        );
      }
    }

    var total = entries.reduce(function (a, e) {
      return a + e.number;
    }, 0);
    if (!(total > 0)) return refuse("the sizes in the paste sum to zero.", { unreadable: unreadable });

    var notes = [];
    if (mode === "percent" || mode === "bare") {
      if (Math.abs(total - 100) > 0.05) {
        notes.push(
          "your percentages sum to " +
            total.toFixed(2) +
            ", not 100 — every number below is your figure divided by " +
            total.toFixed(2)
        );
      }
    }
    var weightBasis =
      mode === "amount"
        ? "your stated amounts, totalling " + total.toLocaleString()
        : "your stated percentages, normalised from " + total.toFixed(2) + " to 100";

    // --- classify every line: fund we can unpack, fund we cannot, or a stock
    var funds = [];
    var direct = Object.create(null);

    entries.forEach(function (e) {
      var w = e.number / total;
      var cls = self.classifyLine(e.name);

      if (cls.kind === "direct") {
        // A plain stock. Its own weight, unchanged.
        direct[e.name.toUpperCase()] = (direct[e.name.toUpperCase()] || 0) + w;
        return;
      }

      // A fund. An index we hold no constituent list for is as un-unpackable
      // as an active fund, so the index must be present in the data file too —
      // a pattern naming "NIFTY SMALLCAP 250" matches but cannot be unpacked.
      var haveList = cls.index && self.d.indices[cls.index];
      funds.push({
        label: cls.label || e.name,
        weight: w,
        index: haveList ? cls.index : null,
        unpacked: !!haveList,
        reason: haveList
          ? cls.why || ""
          : cls.index
            ? '"' + (cls.label || e.name) + '" tracks ' + cls.index +
              ", which is not one of the " + Object.keys(self.d.indices).length +
              " indices this page holds a published constituent list for, so it " +
              "was left un-unpacked rather than mapped to a nearby index."
            : cls.why || self.opaqueFallbackWhy,
        n_constituents: haveList ? self.d.indices[cls.index].n_constituents : 0,
      });
    });

    // --- the arithmetic
    //
    // `listed` is what the reader wrote down; `actual` is what they hold. A
    // symbol present in `actual` but ABSENT from `listed` is the interesting
    // case — a company reached only through a fund — and it must stay absent
    // rather than become 0, because "0%" and "you never mentioned it" are
    // different claims and only one of them is true.
    var actual = Object.create(null);
    var via = Object.create(null);
    var listed = Object.create(null);
    Object.keys(direct).forEach(function (s) {
      listed[s] = direct[s];
      actual[s] = direct[s];
    });

    funds.forEach(function (f) {
      /* The fund's own line keeps the label VERBATIM, not upper-cased. The
         server does the same, and it matters: "Parag Parikh Flexi Cap Fund"
         shouted back as "PARAG PARIKH FLEXI CAP FUND" reads like a ticker the
         reader does not recognise, in a table where every other upper-case row
         really is one. */
      var fsym = f.label;
      if (!f.unpacked) {
        // Counted, never dropped: a fund silently left out is a wrong total.
        // It keeps its own weight, as its own line, listed and actual equal.
        listed[fsym] = (listed[fsym] || 0) + f.weight;
        actual[fsym] = (actual[fsym] || 0) + f.weight;
        return;
      }
      var ix = self.d.indices[f.index];
      Object.keys(ix.weights).forEach(function (sym) {
        var cw = ix.weights[sym];
        var add = f.weight * cw;
        if (!(add > 0)) return;
        actual[sym] = (actual[sym] || 0) + add;
        (via[sym] = via[sym] || []).push({
          source: f.label,
          index: f.index,
          fund_weight: f.weight,
          constituent_weight: cw,
          weight: add,
        });
      });
    });

    var sum = Object.keys(actual).reduce(function (a, s) {
      return a + actual[s];
    }, 0);
    if (Math.abs(sum - 1) > 1e-6) {
      // Refuse rather than render. Every figure on the page is a share of this
      // total, so a total that is not 1 makes all of them quietly wrong.
      return refuse(
        "internal check failed: the looked-through weights sum to " +
          sum.toFixed(6) +
          ", not 1. No number is shown rather than a wrong one.",
        { unreadable: unreadable }
      );
    }

    var opaqueSyms = Object.create(null);
    funds.forEach(function (f) {
      if (!f.unpacked) opaqueSyms[f.label] = true;
    });

    var lines = Object.keys(actual).map(function (sym) {
      var wasListed = Object.prototype.hasOwnProperty.call(listed, sym);
      return {
        symbol: sym,
        name: self._nameOf(sym),
        listed: wasListed ? listed[sym] : null,
        actual: actual[sym],
        kind: opaqueSyms[sym] ? "opaque_fund" : "company",
        via: via[sym] || [],
      };
    });

    // Ranked by SURPRISE — how much of the holding the reader did not list —
    // not by size. A 2% position they never listed is more interesting than a
    // large one that moved 0.1pp.
    lines.forEach(function (l) {
      l.added = l.actual - (l.listed || 0);
    });
    lines.sort(function (a, b) {
      return b.added - a.added;
    });

    var companies = lines.filter(function (l) {
      return l.kind === "company";
    });
    var nUnpacked = funds.filter(function (f) {
      return f.unpacked;
    }).length;

    var top3 = lines
      .slice()
      .sort(function (a, b) {
        return b.actual - a.actual;
      })
      .slice(0, 3)
      .reduce(function (a, l) {
        return a + l.actual;
      }, 0);

    var biggest = lines.length && lines[0].added > 0 ? lines[0] : null;

    return {
      computed: true,
      n_listed: nListed,
      n_unpacked: nUnpacked,
      n_companies: companies.length,
      headline:
        "You listed " +
        nListed +
        " position" +
        (nListed === 1 ? "" : "s") +
        ". Looking through " +
        nUnpacked +
        " of them, you hold " +
        companies.length +
        " companies.",
      biggest_surprise: biggest
        ? { symbol: biggest.symbol, listed: biggest.listed, actual: biggest.actual, added: biggest.added }
        : null,
      top3_share: top3,
      rows: lines,
      funds: funds,
      residual: 0,
      weight_basis: weightBasis,
      weight_source: "scripts/export_lookthrough_data.py, from agent.lookthrough.indices",
      notes: notes,
      unreadable: unreadable,
      approximation_notes: this._approxNotes(funds),
    };
  };

  Engine.prototype._nameOf = function (sym) {
    var ix = this.d.indices;
    for (var k in ix) {
      if (ix[k].names && ix[k].names[sym]) return ix[k].names[sym];
    }
    return "";
  };

  /* The approximation, carried to wherever the number is shown rather than a
     footer. Index weights are computed from free-float market cap because NSE
     publishes no free weights; a reader who discovers that later would be
     right to distrust everything above it. */
  Engine.prototype._approxNotes = function (funds) {
    var out = [];
    var seen = Object.create(null);
    var self = this;
    funds.forEach(function (f) {
      if (!f.unpacked || seen[f.index]) return;
      seen[f.index] = true;
      var ix = self.d.indices[f.index];
      if (ix && ix.accuracy_note) out.push(f.index + ": " + ix.accuracy_note);
    });
    return out;
  };

  // ---------------------------------------------------------------------

  return {
    Engine: Engine,
    // Exported for the parity test, which drives them directly over the
    // Python-generated corpus.
    fold: fold,
    nfkc: nfkc,
    splitGluedDigits: splitGluedDigits,
    stripExchangeSuffix: stripExchangeSuffix,
  };
});
