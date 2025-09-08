import argparse
import glob
import time
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import requests

API_KEY            = ""
API_SECRET_KEY     = ""
BEARER_TOKEN       = ""
ACCESS_TOKEN       = ""
ACCESS_TOKEN_SECRET= ""

def parse_datetime_safe(s):
    if pd.isna(s):
        return pd.NaT
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def ensure_bool(col: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(col):
        return col
    c = col.copy()
    c = c.replace({"True": True, "False": False, "true": True, "false": False, "1": 1, "0": 0})
    try:
        return c.astype(bool)
    except Exception:
        return c.apply(lambda x: bool(x) and not pd.isna(x))

def safe_div(num, den):
    return float(num) / float(den) if (den and den != 0) else np.nan

def detect_columns(df: pd.DataFrame, overrides: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    def pick(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    cols = {}
    cols["user"] = overrides.get("user_col") or pick(["username","user_name","user","screen_name","user_id"])
    if cols["user"] is None:
        raise ValueError("User column not found. Provide --user-col or include one of: username, user_name, user, screen_name, user_id")
    cols["ts"] = overrides.get("time_col") or pick(["tweet_created_at","created_at","timestamp","time"])
    if cols["ts"] is None:
        raise ValueError("Timestamp column not found. Provide --time-col or include one of: tweet_created_at, created_at, timestamp, time")
    cols["rt"] = overrides.get("retweet_flag") or pick(["is_retweet","retweeted","is_rt","retweeted_status"])
    cols["quote"] = overrides.get("quote_flag") or pick(["is_quote","is_quote_status","quoted","is_quoted","referenced_tweets.type_quoted"])
    cols["reply"] = overrides.get("reply_flag") or pick(["is_reply","replied","in_reply_to_user_id","in_reply_to_status_id","referenced_tweets.type_replied_to"])
    cols["rt_count"] = overrides.get("retweet_count") or pick(["retweet_count","retweets_count","public_metrics.retweet_count"])
    cols["rp_count"] = overrides.get("reply_count") or pick(["reply_count","replies_count","public_metrics.reply_count"])
    cols["qt_count"] = overrides.get("quote_count") or pick(["quote_count","quotes_count","public_metrics.quote_count"])
    cols["text"] = overrides.get("text_col") or pick(["tweet_text","text","full_text"])
    cols["csv_followers"] = pick(["followers_count","follower_count","followers"])
    cols["csv_following"] = pick(["following_count","friends_count","followees"])
    cols["csv_tweet_total"] = pick(["tweet_count","statuses_count","total_posts"])
    cols["csv_listed"] = pick(["listed_count","listed_num"])
    cols["csv_age_years"] = pick(["account_age_years","age_years"])
    cols["csv_age_days"] = pick(["account_age_days","age_days"])
    return cols

def build_flags(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    if cols["rt"] is not None:
        rt = ensure_bool(df[cols["rt"]])
    else:
        if cols["text"] is None:
            rt = pd.Series(False, index=df.index)
        else:
            rt = df[cols["text"]].fillna("").str.startswith("RT @").astype(bool)
    if cols["quote"] is not None and cols["quote"] in df.columns:
        q_col = df[cols["quote"]]
        q = q_col if pd.api.types.is_bool_dtype(q_col) else q_col.fillna("").astype(str).str.lower().isin({"1","true","quoted"})
    else:
        q = pd.Series(False, index=df.index)
    if cols["reply"] is not None and cols["reply"] in df.columns:
        rp_col = df[cols["reply"]]
        if pd.api.types.is_bool_dtype(rp_col):
            rp = rp_col
        else:
            if pd.api.types.is_numeric_dtype(rp_col):
                rp = rp_col.fillna(0) != 0
            else:
                rp = ~rp_col.isna()
                rp = rp | rp_col.astype(str).str.lower().eq("replied_to")
    else:
        rp = pd.Series(False, index=df.index)
    orig = ~(rt | q | rp)
    return pd.DataFrame({"_is_original": orig, "_is_retweet": rt, "_is_quote": q, "_is_reply": rp}, index=df.index)

def take_last_n(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    sub_sorted = sub.sort_values("_ts_parsed", ascending=True, na_position="first", kind="mergesort")
    return sub_sorted.tail(n)

def derive_hu_features_per_user(df: pd.DataFrame, cols: Dict[str, Optional[str]], window_size: int, rates_on_originals: bool = False) -> pd.DataFrame:
    ts = df[cols["ts"]].apply(parse_datetime_safe)
    df = df.copy()
    df["_ts_parsed"] = ts
    flags = build_flags(df, cols)
    df = pd.concat([df, flags], axis=1)
    rt_count = df[cols["rt_count"]] if cols["rt_count"] in df.columns else pd.Series(np.nan, index=df.index)
    rp_count = df[cols["rp_count"]] if cols["rp_count"] in df.columns else pd.Series(np.nan, index=df.index)
    qt_count = df[cols["qt_count"]] if cols["qt_count"] in df.columns else pd.Series(np.nan, index=df.index)
    g = df.groupby(cols["user"], dropna=False)
    rows = []
    for user, sub in g:
        win = take_last_n(sub, window_size)
        N = int(len(win))
        n_rt = int(win["_is_retweet"].sum())
        n_q = int(win["_is_quote"].sum())
        n_rp = int(win["_is_reply"].sum())
        n_orig = int(win["_is_original"].sum())
        pct_rt = safe_div(n_rt, N)
        pct_q = safe_div(n_q, N)
        pct_rp = safe_div(n_rp, N)
        pct_orig = safe_div(n_orig, N)
        interactive_per = safe_div(n_rt + n_q + n_rp, N)
        win_sorted = win.sort_values("_ts_parsed", ascending=True, na_position="last")
        diffs = win_sorted["_ts_parsed"].diff().dropna().dt.total_seconds() / (3600 * 24)
        avg_interval_days = float(diffs.mean()) if len(diffs) > 0 else np.nan
        if rates_on_originals:
            mask = win["_is_original"]
            denom = int(mask.sum())
            mean_retweeted = float(win.loc[mask, rt_count.name].astype(float).mean()) if denom > 0 and rt_count.name in win else np.nan
            mean_quoted = float(win.loc[mask, qt_count.name].astype(float).mean()) if denom > 0 and qt_count.name in win else np.nan
            mean_replied = float(win.loc[mask, rp_count.name].astype(float).mean()) if denom > 0 and rp_count.name in win else np.nan
        else:
            mean_retweeted = float(win[rt_count.name].astype(float).mean()) if rt_count.name in win else np.nan
            mean_quoted = float(win[qt_count.name].astype(float).mean()) if qt_count.name in win else np.nan
            mean_replied = float(win[rp_count.name].astype(float).mean()) if rp_count.name in win else np.nan
        rows.append({
            "user": user,
            "HU_R_TweetNum": N,
            "HU_R_TweetPercent_Original": pct_orig,
            "HU_R_TweetPercent_Retweet": pct_rt,
            "HU_R_TweetPercent_Quote": pct_q,
            "HU_R_TweetPercent_Reply": pct_rp,
            "HU_R_RetweetPercent": pct_rt,
            "HU_R_QuotePercent": pct_q,
            "HU_R_ReplyPercent": pct_rp,
            "HU_R_InteractivePer": interactive_per,
            "HU_R_AverageInterval_days": avg_interval_days,
            "HU_R_RetweetedRate": mean_retweeted,
            "HU_R_QuotedRate": mean_quoted,
            "HU_R_RepliedRate": mean_replied
        })
    features = pd.DataFrame(rows)
    cols_order = [
        "user","HU_R_TweetNum","HU_R_TweetPercent_Original","HU_R_TweetPercent_Retweet",
        "HU_R_TweetPercent_Quote","HU_R_TweetPercent_Reply","HU_R_RetweetPercent",
        "HU_R_QuotePercent","HU_R_ReplyPercent","HU_R_InteractivePer","HU_R_AverageInterval_days",
        "HU_R_RetweetedRate","HU_R_QuotedRate","HU_R_RepliedRate"
    ]
    return features.reindex(columns=cols_order)

def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _req_get(url, headers, params, max_retries=3):
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(60 if attempt == 0 else 120); continue
        if r.status_code >= 500:
            time.sleep(5); continue
        r.raise_for_status(); return r
    r.raise_for_status(); return r

def fetch_user_profiles_from_api(user_keys: List[str], id_type: str, bearer_token: str) -> pd.DataFrame:
    if not bearer_token or len(user_keys) == 0:
        return pd.DataFrame(columns=["key","username","id","created_at","followers_count","following_count","tweet_count","listed_count","has_profile_url"])
    headers = {"Authorization": f"Bearer {bearer_token}"}
    fields = "public_metrics,created_at,entities,url,verified,protected"
    rows = []
    if id_type == "user_id":
        base_url = "https://api.twitter.com/2/users"
        for chunk in chunked(user_keys, 100):
            params = {"ids": ",".join([str(x) for x in chunk]), "user.fields": fields}
            r = _req_get(base_url, headers, params); data = r.json().get("data", [])
            for u in data:
                pm = u.get("public_metrics", {}); ent = u.get("entities", {}); url_obj = ent.get("url", {})
                has_url = 1 if (u.get("url") or (isinstance(url_obj.get("urls"), list) and len(url_obj["urls"])>0)) else 0
                rows.append({"key": str(u.get("id")), "username": u.get("username"), "id": str(u.get("id")),
                             "created_at": u.get("created_at"),
                             "followers_count": pm.get("followers_count"), "following_count": pm.get("following_count"),
                             "tweet_count": pm.get("tweet_count"), "listed_count": pm.get("listed_count"),
                             "has_profile_url": has_url})
    else:
        base_url = "https://api.twitter.com/2/users/by"
        cleaned = [str(k).lstrip("@") for k in user_keys]
        for chunk in chunked(cleaned, 100):
            params = {"usernames": ",".join(chunk), "user.fields": fields}
            r = _req_get(base_url, headers, params); data = r.json().get("data", [])
            for u in data:
                pm = u.get("public_metrics", {}); ent = u.get("entities", {}); url_obj = ent.get("url", {})
                has_url = 1 if (u.get("url") or (isinstance(url_obj.get("urls"), list) and len(url_obj["urls"])>0)) else 0
                rows.append({"key": u.get("username"), "username": u.get("username"), "id": str(u.get("id")),
                             "created_at": u.get("created_at"),
                             "followers_count": pm.get("followers_count"), "following_count": pm.get("following_count"),
                             "tweet_count": pm.get("tweet_count"), "listed_count": pm.get("listed_count"),
                             "has_profile_url": has_url})
    return pd.DataFrame(rows)

def compute_profile_features(profile_df: pd.DataFrame, now_ts: pd.Timestamp) -> pd.DataFrame:
    out = profile_df.copy()
    out["created_at_parsed"] = out["created_at"].apply(parse_datetime_safe)
    age_days = (now_ts - out["created_at_parsed"]).dt.total_seconds() / (3600*24)
    out["U_R_AccountAge_days"] = age_days
    out["U_R_ListedNum"] = pd.to_numeric(out["listed_count"], errors="coerce")
    out["U_R_ProfileUrl"] = pd.to_numeric(out["has_profile_url"], errors="coerce").fillna(0).astype(int)
    followers = pd.to_numeric(out["followers_count"], errors="coerce")
    following = pd.to_numeric(out["following_count"], errors="coerce")
    tweets_total = pd.to_numeric(out["tweet_count"], errors="coerce")
    listed = pd.to_numeric(out["listed_count"], errors="coerce")
    out["U_R_FollowerNumDay"] = followers / out["U_R_AccountAge_days"]
    out["U_R_FolloweeNumDay"] = following / out["U_R_AccountAge_days"]
    out["U_R_TweetNumDay"] = tweets_total / out["U_R_AccountAge_days"]
    out["U_R_ListedNumDay"] = listed / out["U_R_AccountAge_days"]
    keep = ["key","username","id","U_R_AccountAge_days","U_R_ListedNum","U_R_ProfileUrl",
            "U_R_FollowerNumDay","U_R_FolloweeNumDay","U_R_TweetNumDay","U_R_ListedNumDay",
            "followers_count","following_count","tweet_count"]
    return out[keep]

def profile_fallback_from_csv(df_users: pd.DataFrame, cols: Dict[str, Optional[str]], now_ts: pd.Timestamp) -> pd.DataFrame:
    out = pd.DataFrame({"key": df_users["user"]})
    followers = pd.to_numeric(df_users[cols["csv_followers"]], errors="coerce") if cols["csv_followers"] else np.nan
    following = pd.to_numeric(df_users[cols["csv_following"]], errors="coerce") if cols["csv_following"] else np.nan
    tweets_total = pd.to_numeric(df_users[cols["csv_tweet_total"]], errors="coerce") if cols["csv_tweet_total"] else np.nan
    listed = pd.to_numeric(df_users[cols["csv_listed"]], errors="coerce") if cols["csv_listed"] else np.nan
    if cols["csv_age_days"] and cols["csv_age_days"] in df_users.columns:
        age_days = pd.to_numeric(df_users[cols["csv_age_days"]], errors="coerce")
    elif cols["csv_age_years"] and cols["csv_age_years"] in df_users.columns:
        age_years = pd.to_numeric(df_users[cols["csv_age_years"]], errors="coerce")
        age_days = age_years * 365.25
    else:
        age_days = pd.Series(np.nan, index=df_users.index)
    out["U_R_AccountAge_days"] = age_days
    out["U_R_ListedNum"] = listed
    out["U_R_ProfileUrl"] = np.nan
    out["U_R_FollowerNumDay"] = followers / age_days
    out["U_R_FolloweeNumDay"] = following / age_days
    out["U_R_TweetNumDay"] = tweets_total / age_days
    out["U_R_ListedNumDay"] = listed / age_days
    out["followers_count"] = followers
    out["following_count"] = following
    out["tweet_count"] = tweets_total
    out["username"] = np.nan
    out["id"] = np.nan
    return out

def main():
    ap = argparse.ArgumentParser(description="Derive HU features (last-N) + profile features via X API")
    ap.add_argument("input", help="Input CSV (or glob pattern)")
    ap.add_argument("output", help="Output CSV path")
    ap.add_argument("--window-size", type=int, default=100, help="N: number of most recent tweets per user to use")
    ap.add_argument("--rates-on-originals", action="store_true", help="Compute HU_R_*Rate over originals only")
    ap.add_argument("--user-col", dest="user_col", default=None)
    ap.add_argument("--time-col", dest="time_col", default=None)
    ap.add_argument("--retweet-flag", dest="retweet_flag", default=None)
    ap.add_argument("--quote-flag", dest="quote_flag", default=None)
    ap.add_argument("--reply-flag", dest="reply_flag", default=None)
    ap.add_argument("--retweet-count", dest="retweet_count", default=None)
    ap.add_argument("--quote-count", dest="quote_count", default=None)
    ap.add_argument("--reply-count", dest="reply_count", default=None)
    ap.add_argument("--text-col", dest="text_col", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input)) or [args.input]
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    overrides = {"user_col": args.user_col,"time_col": args.time_col,"retweet_flag": args.retweet_flag,
                 "quote_flag": args.quote_flag,"reply_flag": args.reply_flag,"retweet_count": args.retweet_count,
                 "quote_count": args.quote_count,"reply_count": args.reply_count,"text_col": args.text_col}
    cols = detect_columns(df, overrides)

    hu = derive_hu_features_per_user(df, cols, window_size=args.window_size, rates_on_originals=args.rates_on_originals)

    now_ts = pd.Timestamp.now(tz="UTC")
    unique_keys = sorted(hu["user"].astype(str).unique().tolist())
    id_type = "user_id" if cols["user"] == "user_id" else "username"

    api_profiles = fetch_user_profiles_from_api(unique_keys, id_type, BEARER_TOKEN)
    if api_profiles.empty:
        tmp = hu[["user"]].copy()
        profile_df = profile_fallback_from_csv(tmp, cols, now_ts)
    else:
        profile_df = compute_profile_features(api_profiles, now_ts)

    profile_df = profile_df.rename(columns={"key": "user"})
    out = hu.merge(profile_df, on="user", how="left")

    final_cols = [
        "user",
        "HU_R_TweetNum","HU_R_TweetPercent_Original","HU_R_TweetPercent_Retweet",
        "HU_R_TweetPercent_Quote","HU_R_TweetPercent_Reply","HU_R_RetweetPercent",
        "HU_R_QuotePercent","HU_R_ReplyPercent","HU_R_InteractivePer","HU_R_AverageInterval_days",
        "HU_R_RetweetedRate","HU_R_QuotedRate","HU_R_RepliedRate",
        "U_R_AccountAge_days","U_R_ListedNum","U_R_ProfileUrl",
        "U_R_FollowerNumDay","U_R_FolloweeNumDay","U_R_TweetNumDay","U_R_ListedNumDay",
        "followers_count","following_count","tweet_count","username","id"
    ]
    for c in final_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out.reindex(columns=final_cols)

    out.to_csv(args.output, index=False)
    print(f"[OK] Window size = {args.window_size}. Users = {len(out)}. API used: {not api_profiles.empty}")

if __name__ == "__main__":
    main()
