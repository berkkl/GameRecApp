---
version: 1
name: GameRecApp
lang: en
description: Neutral surfaces, a petrol-blue accent and readable cards for Steam game discovery.
colors:
  background: "#fafafa"
  surface: "#ffffff"
  surface-muted: "#f2f4f5"
  text: "#24272b"
  text-muted: "#616970"
  primary: "#185c68"
  primary-hover: "#124954"
  primary-soft: "#eaf3f5"
  border: "#e3e6e8"
  border-control: "#879399"
  success: "#266344"
  success-soft: "#eef6f0"
  warning: "#86531c"
  warning-soft: "#fbf3e6"
  danger: "#a13232"
  danger-soft: "#fef1f1"
typography:
  body: "16px / 1.65"
  interface: "14px / 1.5"
  metadata: "13px"
  page-title: "clamp(28px, 3vw, 40px)"
  section-title: "24px"
  card-title: "20px"
spacing:
  1: "4px"
  2: "8px"
  3: "12px"
  4: "16px"
  5: "20px"
  6: "24px"
  8: "32px"
  10: "40px"
rounded:
  control: "8px"
  panel: "12px"
---

# GameRecApp design system

People describe a game they want, add games they enjoy, and explore Steam recommendations. Search and results stay on the same page.

## Stylesheet ownership

- `public/design.css`: shared colors, 4px spacing scale, typography, corners, inputs and keyboard focus. Discovery, history and generated benchmark pages use these tokens.
- `public/discover.css`: discovery layout, search form, starting points and result cards.
- `public/sequel.css`: related-game disclosures.
- `public/index.html`: discovery markup. `public/app.js` renders result cards.
- `public/history.html`: page layout using the shared colors. History is available only when enabled on the server. Generated benchmark pages follow the same convention.

New styles reuse the shared tokens instead of redefining them.

## Colors and surfaces

Use `#FAFAFA` for the page, white for cards and `#F2F4F5` for supporting areas. Use `#185C68` for primary actions, branding, links and keyboard focus. Selected options use `#EAF3F5`. Avoid large decorative areas filled with the accent color.

Primary text is `#24272B`; supporting text is `#616970`. Card borders use `#E3E6E8`; inputs and secondary buttons use `#879399`. Success, warning and error states use green, brown and red text on pale backgrounds. Always include text such as "Saved", an error explanation or a benchmark evaluation label; color alone must not convey state.

Game artwork keeps its original colors. Starting-point titles appear on white beneath the image, without lengthy overlays or dark gradients. If artwork is unavailable, show the game name.

## Typography and language

Use `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif` without external font requests. Body text is 16px/1.65, interface text 14px/1.5 and metadata 13px. Headings use weight 600: page titles 28-40px, section titles 24px and game titles 20px.

The interface and repository documentation are English. Queries can be multilingual; this does not imply a translated interface. Use action labels such as "Games you love", "More filters", "View on Steam" and "Add to taste". Avoid motivational slogans. Long game names, queries and button labels wrap.

## Layout

The discovery shell is at most 1440px wide with 40px desktop margins. The left column is a 352px search and preferences form, separated from results by 32px. At widths of 900px or less, use 24px margins and a 320px form.

At 760px or less, stack the form above results with 16px page margins. Form fields may use two columns at intermediate widths, returning to one below 600px. Controls remain in normal page flow, without an independently scrolling or fixed-height form. The desktop top bar is sticky; on narrow screens it returns to normal flow.

Results use two columns when space permits, becoming one below 1100px while the form remains beside them. Once the form is above the results, two result columns may fit between 600px and 760px. Card padding is 24px on desktop and 20px on mobile.

Use 8-12px within groups, 24px between groups and 32px between major sections. Leave 8px between a label and input. Prefer CSS logical properties. Wide history and benchmark tables scroll inside their own containers without overflowing the page.

## Controls and cards

Each task group has one primary action: "Find my next game". Primary buttons use white text on the accent color. Secondary buttons use white backgrounds and visible borders. Buttons and inputs are at least 44px high. Input text is at least 16px, with 8px corners and persistent visible labels.

Cards use a white background, 1px surface border and 12px corners. Ordinary cards have no shadow or hover motion. Indicate hover through color or border changes. Add, remove and feedback controls may wrap.

Filters, matching text, related games and feedback use native `details`/`summary` disclosures with visible indicators. Interactive elements receive a 2px accent outline on keyboard focus. Radio-option labels carry the focus outline. Disable animation and transitions when reduced motion is requested.

## States and data

While searching, show loading text and neutral skeleton cards. Empty results offer a way to change the query or clear filters. Errors explain the problem and an available next action. Successful game selection shows a brief status message.

Preserve catalog coverage and freshness notices. Keep Steam links, artwork, review ratios, play modes and matching source text in result cards. Clearly label synthetic demo data. Design consistency does not guarantee data accuracy or recommendation quality.

## Verification

Check new layouts at widths of 320, 390, 760, 900, 1280 and 1440px. Include 200% zoom, long multilingual queries and titles, keyboard focus, filter disclosures, game selection, results, loading and errors. RTL is a layout robustness check, not a claim of additional language support. Report checks that were not performed.
