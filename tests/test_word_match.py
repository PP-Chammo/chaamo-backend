import os
import sys
import types
import pytest

# Ensure the repository root is importable so 'src' package can be resolved
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Stub src.utils.supabase to avoid real DB/network
supabase_stub_mod = types.ModuleType('src.utils.supabase')
supabase_stub_mod.supabase = types.SimpleNamespace()
sys.modules['src.utils.supabase'] = supabase_stub_mod

# Stub src.utils.logger to avoid logger setup side effects
logger_stub_mod = types.ModuleType('src.utils.logger')
logger_stub = types.SimpleNamespace(
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
)
logger_stub_mod.scraper_logger = logger_stub
logger_stub_mod.setup_logger = lambda *a, **k: logger_stub
sys.modules['src.utils.logger'] = logger_stub_mod

from src.utils.ebay import word_match


# ---------- Simple and baseline behavior ----------

def test_empty_filter_returns_true():
    assert word_match("any title", "") is True
    assert word_match("any title", None) is True


def test_none_title_with_nonempty_filter_returns_false():
    assert word_match(None, "topps") is False


def test_exact_token_presence_meets_threshold():
    title = "2021 Topps Chrome UEFA 14 Cristiano Ronaldo PSA 10"
    assert word_match(title, "topps chrome", word_threshold=2) is True


def test_non_strict_tokens_allow_fuzzy_but_not_too_far():
    title = "2021 Topps Chrome UEFA Champions Lg Card"
    # 'league' -> 'lg' is too far for partial_ratio at 80; should fail to hit threshold
    assert (
        word_match(
            title,
            "topps chrome league",
            word_threshold=3,
        )
        is False
    )


def test_threshold_not_met_is_false_but_true_if_filter_shorter():
    base_title = "2021 Topps Chrome 14 Cristiano Ronaldo PSA 10"
    # Needs 3 matches, we only have 'topps' and 'chrome' => False
    assert word_match(base_title, "topps chrome uefa champions league", word_threshold=3) is False
    # If filter is only the two strict tokens, threshold 2 is met => True
    assert word_match(base_title, "topps chrome", word_threshold=2) is True


# ---------- Aliases and normalization ----------

def test_logo_fractor_collapsed_true():
    title = "Topps Chrome Logofractor Parallel"
    # 'logo fractor' should be detected via collapsed/squeezed variants
    assert word_match(title, "logo fractor", word_threshold=1) is True


def test_logo_fractor_with_hyphen_and_case_variants():
    # Ensure hyphenated and case-insensitive matching works
    title = "TOPPS CHROME LOGO-FRACTOR PARALLEL"
    assert word_match(title, "logo fractor", word_threshold=1) is True


# ---------- Fuzzy matching for non-strict tokens ----------

def test_fuzzy_for_non_strict_token_allows_minor_typos():
    # Keep strict tokens exact, allow fuzzy on later token
    title = "2021 Topps Chrome UEFa Champions Leegue Card"
    # Strictly require 'topps' and 'chrome'; allow fuzzy to satisfy 'league' later
    assert (
        word_match(
            title,
            "topps chrome league",
            word_threshold=3,
        )
        is True
    )


def test_perfect_whole_string_equality():
    title = "Topps Chrome"
    # Require all tokens; both present; perfect equality path can trigger
    assert word_match(title, "topps chrome", word_threshold=None) is True


# ---------- Token merging (single-char) ----------

def test_single_char_token_match():
    title = "2014 Spiderman-1 Marvel Card"
    # Filter tokens split; both must match
    assert word_match(title, "spiderman 1", word_threshold=2) is True


def test_hyphenated_vs_compact_player_names():
    title = "Cristiano-Ronaldo Topps Chrome"
    assert word_match(title, "cristiano ronaldo chrome", word_threshold=3) is True


# ---------- Edge cases ----------

def test_title_without_required_tokens_fails():
    title = "Panini Prizm UEFA"
    assert word_match(title, "topps chrome uefa", word_threshold=2) is False


def test_no_strict_required_threshold_all_tokens():
    title = "Topps Chrome UEFA Champions League"
    # Require all tokens but no strict; should succeed
    assert word_match(title, "topps chrome uefa", word_threshold=None) is True


def test_case_insensitivity_and_spaces_irrelevant():
    title = "   topps   CHROME   uefa   "
    assert word_match(title, "Topps Chrome UEFA", word_threshold=3) is True


# ---------- New scenarios ----------

def test_paldean_fates_partial_match():
    title = "PICK YOUR CARD! Scarlet & Violet: Paldean Fates - Common/Uncommon/Rare Holo"
    flt = "Pokemon Scarlet & Violet Paldean Fates"
    assert word_match(title, flt, word_threshold=2) is True
