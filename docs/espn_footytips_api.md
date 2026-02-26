# ESPN Footytips API — Complete Mapping

## Overview

ESPN Footytips uses a REST API at `https://api.footytips.espn.com.au` with Disney OneID (OAuth2) authentication. The frontend is a React SPA that communicates with this API via Axios.

---

## Authentication

### Disney OneID (OAuth2)

The site uses Disney's OneID SDK loaded from:
```
https://cdn.registerdisney.go.com/v4/OneID.js
```

**Config passed to OneID:**
```json
{
  "clientId": "ESPN-FOOTYTIPS.WEB",
  "responderPage": "https://footytips.espn.com.au/responder.html"
}
```

**Auth flow:**
1. OneID SDK initializes → checks if user has a valid Disney session
2. On login, OneID returns `{ token: { access_token: "..." }, profile: { firstName, lastName, email, swid, ... } }`
3. All API calls use header: `Authorization: Bearer <access_token>`
4. On 401 responses, the app silently refreshes the token via `OneID.get()` and retries

**For automation**, you need to:
1. Log in once via the browser to get an `access_token`
2. Store it and use in `Authorization: Bearer <token>` header
3. Refresh it when it expires (token refresh returns a new `access_token`)

---

## Key Constants

| Constant | Value |
|---|---|
| Base URL | `https://api.footytips.espn.com.au` |
| Affiliate ID | `1` |
| Sport slug (NRL) | `rugby-league` |
| Sport ID (NRL) | `3` |
| League slug (NRL) | `nrl` |
| League ID (NRL) | `2` |
| Client ID | `ESPN-FOOTYTIPS.WEB` |
| Competition ID | `656543` (your comp) |
| Game Type | `tipping` |

---

## NRL Team ID Mapping

| Team | teamId | shortCode |
|---|---|---|
| Broncos | 1 | BRIS |
| Bulldogs | 2 | BULL |
| Raiders | 3 | CANB |
| Storm | 4 | MELB |
| Knights | 5 | NEWC |
| Sea Eagles | 6 | MANL |
| Cowboys | 7 | NQLD |
| Eels | 8 | PARR |
| Panthers | 9 | PENR |
| Roosters | 10 | SYDR |
| Sharks | 11 | SHRK |
| Dragons | 12 | DRAG |
| NZ Warriors | 14 | NZW |
| Wests Tigers | 15 | WTIG |
| Rabbitohs | 45 | SSYD |
| Titans | 509 | TITN |
| Dolphins | 1706 | DOL |

---

## API Endpoints

### 1. Get Sports & Leagues (Public — no auth required)

```
GET /clients/1/sports/leagues?includeGameTypes=true&includeTeams=true
```

Returns all sports, leagues, teams, current round info, and season status.

---

### 2. Get Current Round Events/Tips (Auth required)

**Personal tips page:**
```
GET /games/sports/{sport}/leagues/{league}/game-types/tipping/rounds/{round}
```

Example:
```
GET /games/sports/rugby-league/leagues/nrl/game-types/tipping/rounds/5
Authorization: Bearer <token>
```

**Response includes:**
- `events[]` — array of matches with `eventId`, `competitors[]` (home/away teams with `teamId`), `dateTime`, `venue`, `eventStatus`, `marginRequired`
- User's existing `tips[]` — `{ eventId, teamId, tipMargin }`
- `odds` data (if available)
- `lockoutDateTime`
- Round metadata

---

### 3. Submit Tips — Personal (Auth required)

```
POST /games/sports/{sport}/leagues/{league}/game-types/tipping/rounds/{round}
Authorization: Bearer <token>
Content-Type: application/json
```

**Request body (`submitTippingTipsRequestSchema`):**
```json
{
  "tips": [
    {
      "eventId": 12345,
      "teamId": 4,
      "dateTime": "Thu Feb 26 2026",
      "round": -1
    },
    {
      "eventId": 12346,
      "teamId": 9,
      "dateTime": "Thu Feb 26 2026",
      "round": -1
    }
  ],
  "integrations": [],
  "clientId": "ESPN-FOOTYTIPS.WEB"
}
```

**Tip object fields:**
| Field | Type | Description |
|---|---|---|
| `eventId` | int | The match/event ID from the GET response |
| `teamId` | int | The team ID you're tipping to win |
| `dateTime` | string | Date string (e.g. `"Thu Feb 26 2026"`) |
| `round` | int | Always `-1` in submissions |
| `tipMargin` | int (optional) | If `marginRequired` is true for the event |

---

### 4. Submit Tips — Competition-Specific (Auth required)

```
POST /competitions/{competitionId}/games/sports/{sport}/leagues/{league}/game-types/{gameType}/rounds/{round}/members/{userId}
Authorization: Bearer <token>
Content-Type: application/json
```

Example for your competition:
```
POST /competitions/656543/games/sports/rugby-league/leagues/nrl/game-types/tipping/rounds/5/members/{userId}
Authorization: Bearer <token>
Content-Type: application/json
```

**Same request body as personal tips:**
```json
{
  "tips": [...],
  "integrations": [],
  "clientId": "ESPN-FOOTYTIPS.WEB"
}
```

**Note:** The `userId` is the Disney SWID (a UUID like `{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}`) from the OneID profile.

---

### 5. Submit Jokers (Auth required)

```
POST /games/sports/{sport}/leagues/{league}/game-types/tipping/rounds/{round}/jokers
Authorization: Bearer <token>
Content-Type: application/json
```

**Request body:**
```json
{
  "jokers": [12345]
}
```

The joker array contains the `eventId` of the match you want to apply your joker/confidence boost to.

---

### 6. Get Last Round Tips

```
GET /games/sports/{sport}/leagues/{league}/game-types/tipping/last-round/{round}
Authorization: Bearer <token>
```

---

### 7. Get Competition Details

```
GET /competitions/{competitionId}?includeLadders=true
Authorization: Bearer <token>
```

Returns competition info, leagues, ladders, members count, admin status.

---

### 8. Get Competition Ladder

```
GET /competitions/{competitionId}/sports/{sport}/leagues/{league}/game-types/{gameType}/ladders/{ladderId}
Authorization: Bearer <token>
```

---

## Automation Flow

### Step-by-step for auto-submitting tips:

```
1. AUTHENTICATE
   - Use stored Disney OneID token
   - Or: login via browser, extract access_token from localStorage/cookies

2. GET ROUND DATA
   GET /games/sports/rugby-league/leagues/nrl/game-types/tipping/rounds/{round}
   → Extract events[] with eventId + competitors (home/away teamId)

3. MAP MODEL PREDICTIONS TO API FORMAT
   For each match:
   - Find the eventId from the API response
   - Map your predicted winner team name → teamId (see team mapping above)
   - Build tip object: { eventId, teamId, dateTime, round: -1 }

4. SUBMIT PERSONAL TIPS
   POST /games/sports/rugby-league/leagues/nrl/game-types/tipping/rounds/{round}
   Body: { tips: [...], integrations: [], clientId: "ESPN-FOOTYTIPS.WEB" }

5. SUBMIT COMPETITION TIPS (for comp 656543)
   POST /competitions/656543/games/sports/rugby-league/leagues/nrl/game-types/tipping/rounds/{round}/members/{userId}
   Body: { tips: [...], integrations: [], clientId: "ESPN-FOOTYTIPS.WEB" }

6. OPTIONALLY SET JOKER
   POST /games/sports/rugby-league/leagues/nrl/game-types/tipping/rounds/{round}/jokers
   Body: { jokers: [eventId_of_most_confident_pick] }
```

### Token Management

```python
# Token is a Disney OneID OAuth2 access_token
# Stored in Redux state as: auth.authorization = "Bearer <access_token>"
# On 401 response: refresh via OneID SDK silentRefresh, get new access_token
# For headless automation: need to implement token refresh or use long-lived session
```

---

## Cache Busting

The app appends a `t=<timestamp>` query parameter to API calls (from `localStorage.bustCache`).

```
GET /games/sports/rugby-league/leagues/nrl/game-types/tipping/rounds/5?t=1709000000000
```

---

## Important Notes

1. **Lockout**: Tips must be submitted before `lockoutDateTime` (first match kickoff of the round)
2. **Event status**: Only events with `eventStatus: "PRE"` can be tipped. Once a match starts, that tip is locked
3. **teamId: -1** means "no tip" — the API filters these out before submission
4. **Margin**: NRL tipping has `marginRequired` on certain events — check the event data
5. **Draw tips**: Some events have `allowDrawTip: true` — use `teamId: 0` for a draw tip
6. **The `round` field in tip objects is always `-1`** — the actual round is in the URL path
7. **Both endpoints needed**: Submit to personal endpoint AND competition endpoint for comp 656543
