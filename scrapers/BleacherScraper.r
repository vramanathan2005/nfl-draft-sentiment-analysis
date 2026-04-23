




library(rvest)
library(xml2)
library(stringr)
library(dplyr)
library(purrr)
library(tibble)
library(furrr)
library(httr2)
library(xml2)
library(rvest)

#-----------------------------
# 1) Get scouting report links
#-----------------------------
get_br_scouting_links <- function(big_board_url) {
  page <- read_html(big_board_url)
  
  links <- page |>
    html_elements("a") |>
    html_attr("href") |>
    unique()
  
  links <- links[!is.na(links)]
  
  # Keep only scouting report article links
  links <- links[str_detect(
    links,
    "bleacherreport\\.com/articles/.+scouting-report"
  )]
  
  tibble(url = unique(links))
}


#---------------------------------------
# 2) Helper: pull text between two labels
#---------------------------------------
get_between <- function(text, start, end) {
  out <- str_match(
    text,
    paste0(start, "\\s*([\\s\\S]*?)\\s*", end)
  )[, 2]
  
  if (is.na(out)) return(NA_character_)
  str_squish(out)
}

#------------------------------------
# 3) Helper: pull bullet-style section
#------------------------------------
get_bullets_between <- function(text, start, end) {
  block <- get_between(text, start, end)
  if (is.na(block)) return(NA_character_)
  
  lines <- str_split(block, "\n")[[1]]
  lines <- str_squish(lines)
  lines <- lines[lines != ""]
  
  # Keep bullet lines beginning with em dash or hyphen-like bullet
  lines <- lines[str_detect(lines, "^—|^-")]
  lines <- str_replace(lines, "^—\\s*|^-\\s*", "")
  
  paste(lines, collapse = " | ")
}

#---------------------------------------
# 4) Scrape one Bleacher Report article
#---------------------------------------
# scrape_br_scouting_report <- function(article_url) {
#   page <- read_html(article_url)
#   
#   txt <- page |>
#     html_element("body") |>
#     html_text2()
#   
#   txt <- str_replace_all(txt, "\r", "\n")
#   txt <- str_replace_all(txt, "[ \t]+", " ")
#   txt <- str_replace_all(txt, "\n{2,}", "\n")
#   txt <- str_squish(txt)
#   
#   # Title
#   title <- str_match(
#     txt,
#     "NFL\\s+(.+?Scouting Report.+?)\\s+BR NFL Scouting Department"
#   )[,2]
#   
#   if (is.na(title)) {
#     title <- page |>
#       html_element("title") |>
#       html_text2() |>
#       str_squish()
#   }
#   
#   # From title
#   page_title <- page |>
#     html_element('meta[property="og:title"]') |>
#     html_attr("content") |>
#     str_squish()
#   
#   player_from_title <- str_match(
#     page_title,
#     "^(.+?)\\s+NFL Draft 2024:"
#   )[,2] |>
#     str_squish()
#   message("Processing: ", player_from_title)
#   
#   report_tail <- str_match(
#     page_title,
#     "Scouting Report for\\s+(.+)$"
#   )[,2] |>
#     str_squish()
#   
#   position_from_title <- str_match(report_tail, "([A-Z]{1,5})$")[,2] |>
#     str_squish()
#   
#   college_from_title <- str_remove(report_tail, "\\s+[A-Z]{1,5}$") |>
#     str_squish()
#   
#   # Byline / date
#   byline <- str_match(txt, "(BR NFL Scouting Department)") [,2]
#   date   <- str_match(txt, "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.?\\s+\\d{1,2},\\s+\\d{4}") [,1]
#   
#   # Measurements
#   height   <- str_match(txt, "HEIGHT:\\s*([^\\n]+?)\\s+WEIGHT:") [,2]
#   weight   <- str_match(txt, "WEIGHT:\\s*([^\\n]+?)\\s+HAND:") [,2]
#   hand     <- str_match(txt, "HAND:\\s*([^\\n]+?)\\s+ARM:") [,2]
#   arm      <- str_match(txt, "ARM:\\s*([^\\n]+?)\\s+WINGSPAN:") [,2]
#   wingspan <- str_match(txt, "WINGSPAN:\\s*([^\\n]+?)\\s+40-YARD DASH:") [,2]
#   dash40   <- str_match(txt, "40-YARD DASH:\\s*([^\\n]*?)\\s+3-CONE:") [,2]
#   cone3    <- str_match(txt, "3-CONE:\\s*([^\\n]*?)\\s+SHUTTLE:") [,2]
#   shuttle  <- str_match(txt, "SHUTTLE:\\s*([^\\n]*?)\\s+VERTICAL:") [,2]
#   vertical <- str_match(txt, "VERTICAL:\\s*([^\\n]*?)\\s+BROAD:") [,2]
#   broad    <- str_match(txt, "BROAD:\\s*([^\\n]*?)\\s+POSITIVES") [,2]
#   
#   # Sections
#   positives <- get_bullets_between(txt, "POSITIVES", "NEGATIVES")
#   negatives <- get_bullets_between(txt, "NEGATIVES", "2023 STATISTICS|2024 STATISTICS|STATISTICS")
#   statistics <- get_bullets_between(txt, "2023 STATISTICS|2024 STATISTICS|STATISTICS", "NOTES")
#   notes <- get_bullets_between(txt, "NOTES", "OVERALL")
#   
#   overall <- get_between(txt, "OVERALL", "GRADE:")
#   
#   grade <- str_match(txt, "GRADE:\\s*([^\\n]+?)\\s+OVERALL RANK:") [,2]
#   overall_rank <- str_match(txt, "OVERALL RANK:\\s*([^\\n]+?)\\s+POSITION RANK:") [,2]
#   position_rank <- str_match(txt, "POSITION RANK:\\s*([^\\n]+?)\\s+PRO COMPARISON:") [,2]
#   pro_comparison <- str_match(txt, "PRO COMPARISON:\\s*([^\\n]+?)\\s+Written by") [,2]
#   written_by <- str_match(txt, "Written by\\s+([^\\n]+)") [,2]
#   
#   tibble(
#     url = article_url,
#     title = str_squish(title),
#     player_from_title = str_squish(player_from_title),
#     position_from_title = str_squish(position_from_title),
#     college_from_title = str_squish(college_from_title),
#     byline = str_squish(byline),
#     date = str_squish(date),
#     height = str_squish(height),
#     weight = str_squish(weight),
#     hand = str_squish(hand),
#     arm = str_squish(arm),
#     wingspan = str_squish(wingspan),
#     dash40 = str_squish(dash40),
#     cone3 = str_squish(cone3),
#     shuttle = str_squish(shuttle),
#     vertical = str_squish(vertical),
#     broad = str_squish(broad),
#     positives = positives,
#     negatives = negatives,
#     statistics = statistics,
#     notes = notes,
#     overall = overall,
#     article_grade = str_squish(grade),
#     overall_rank = str_squish(overall_rank),
#     position_rank = str_squish(position_rank),
#     pro_comparison = str_squish(pro_comparison),
#     written_by = str_squish(written_by)
#   )
# }


library(rvest)
library(stringr)
library(tibble)

scrape_br_scouting_report_faster <- function(article_url) {
  page <- read_html(article_url)
  
  clean1 <- function(x) {
    x <- x[1]
    if (length(x) == 0 || is.na(x)) return(NA_character_)
    str_squish(x)
  }
  
  section_between <- function(lines, start_pat, end_pat) {
    start_idx <- which(str_detect(lines, regex(start_pat, ignore_case = TRUE)))[1]
    end_idx   <- which(str_detect(lines, regex(end_pat, ignore_case = TRUE)))[1]
    
    if (is.na(start_idx) || is.na(end_idx) || end_idx <= start_idx) {
      return(NA_character_)
    }
    
    out <- lines[(start_idx + 1):(end_idx - 1)]
    out <- out[out != ""]
    if (length(out) == 0) return(NA_character_)
    paste(out, collapse = " | ")
  }
  
  text_between <- function(lines, start_pat, end_pat) {
    start_idx <- which(str_detect(lines, regex(start_pat, ignore_case = TRUE)))[1]
    end_idx   <- which(str_detect(lines, regex(end_pat, ignore_case = TRUE)))[1]
    
    if (is.na(start_idx) || is.na(end_idx) || end_idx <= start_idx) {
      return(NA_character_)
    }
    
    out <- lines[(start_idx + 1):(end_idx - 1)]
    out <- out[out != ""]
    if (length(out) == 0) return(NA_character_)
    str_squish(paste(out, collapse = " "))
  }
  
  # fast metadata
  og_title <- page |>
    html_element('meta[property="og:title"]') |>
    html_attr("content") |>
    clean1()
  
  if (is.na(og_title)) {
    og_title <- page |>
      html_element("title") |>
      html_text2() |>
      clean1()
  }
  
  player_from_title <- str_match(
    og_title,
    "^(.+?)\\s+NFL Draft\\s+\\d{4}:"
  )[,2] |>
    clean1()
  
  report_tail <- str_match(
    og_title,
    "Scouting Report for\\s+(.+)$"
  )[,2] |>
    clean1()
  
  position_from_title <- if (!is.na(report_tail)) {
    str_match(report_tail, "([A-Z]{1,5})$")[,2] |>
      clean1()
  } else NA_character_
  
  college_from_title <- if (!is.na(report_tail)) {
    str_remove(report_tail, "\\s+[A-Z]{1,5}$") |>
      clean1()
  } else NA_character_
  
  # only pull likely article text nodes, not whole body
  lines <- page |>
    html_elements("h1, h2, h3, h4, p, li") |>
    html_text2() |>
    str_squish()
  
  lines <- lines[!is.na(lines) & lines != ""]
  
  # compact string only from relevant nodes
  txt <- paste(lines, collapse = "\n")
  
  date <- str_match(
    txt,
    "((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.?\\s+\\d{1,2},\\s+\\d{4})"
  )[,2] |>
    clean1()
  
  # measurements
  height   <- str_match(txt, "HEIGHT:\\s*([^\\n]+?)\\s+WEIGHT:")[,2] |> clean1()
  weight   <- str_match(txt, "WEIGHT:\\s*([^\\n]+?)\\s+HAND:")[,2] |> clean1()
  hand     <- str_match(txt, "HAND:\\s*([^\\n]+?)\\s+ARM:")[,2] |> clean1()
  arm      <- str_match(txt, "ARM:\\s*([^\\n]+?)\\s+WINGSPAN:")[,2] |> clean1()
  wingspan <- str_match(txt, "WINGSPAN:\\s*([^\\n]+?)\\s+40-YARD DASH:")[,2] |> clean1()
  dash40   <- str_match(txt, "40-YARD DASH:\\s*([^\\n]*?)\\s+3-CONE:")[,2] |> clean1()
  cone3    <- str_match(txt, "3-CONE:\\s*([^\\n]*?)\\s+SHUTTLE:")[,2] |> clean1()
  shuttle  <- str_match(txt, "SHUTTLE:\\s*([^\\n]*?)\\s+VERTICAL:")[,2] |> clean1()
  vertical <- str_match(txt, "VERTICAL:\\s*([^\\n]*?)\\s+BROAD:")[,2] |> clean1()
  broad    <- str_match(txt, "BROAD:\\s*([^\\n]*?)\\s+POSITIVES")[,2] |> clean1()
  
  positives <- section_between(lines, "^POSITIVES:?$", "^NEGATIVES:?$")
  
  positives <- if (!is.na(positives)) {
    str_replace(positives, "^.*?(—)", "\\1") |> str_squish()
  } else {
    NA_character_
  }
  
  negatives <- section_between(
    lines,
    "^NEGATIVES:?$",
    "^(\\d{4} STATISTICS:?|STATISTICS:?|NOTES:?)$"
  )
  
  statistics <- section_between(
    lines,
    "^(\\d{4} STATISTICS:?|STATISTICS:?)$",
    "^NOTES:?$"
  )
  
  notes <- section_between(
    lines,
    "^NOTES:?$",
    "^OVERALL:?$"
  )
  
  overall <- text_between(
    lines,
    "^OVERALL:?$",
    "^GRADE:"
  )
  
  article_grade <- str_match(txt, "GRADE:\\s*([^\\n]+?)\\s+OVERALL RANK:")[,2] |> clean1()
  overall_rank  <- str_match(txt, "OVERALL RANK:\\s*([^\\n]+?)\\s+POSITION RANK:")[,2] |> clean1()
  position_rank <- str_match(txt, "POSITION RANK:\\s*([^\\n]+?)\\s+PRO COMPARISON:")[,2] |> clean1()
  pro_comparison <- str_match(txt, "PRO COMPARISON:\\s*([^\\n]+?)\\s+Written by")[,2] |> clean1()
  
  tibble(
    url = article_url,
    title = og_title,
    player_from_title = player_from_title,
    position_from_title = position_from_title,
    college_from_title = college_from_title,
    date = date,
    height = height,
    weight = weight,
    hand = hand,
    arm = arm,
    wingspan = wingspan,
    dash40 = dash40,
    cone3 = cone3,
    shuttle = shuttle,
    vertical = vertical,
    broad = broad,
    positives = positives,
    negatives = negatives,
    statistics = statistics,
    notes = notes,
    overall = overall,
    article_grade = article_grade,
    overall_rank = overall_rank,
    position_rank = position_rank,
    pro_comparison = pro_comparison
  )
}



big_board_url <- "https://bleacherreport.com/articles/10117937-br-nfl-scouting-depts-final-2024-nfl-draft-big-board"

links_tbl <- get_br_scouting_links(big_board_url)

# scrape one
one_report <- scrape_br_scouting_report_faster(links_tbl$url[1])

# scrape all
all_reports <- map_dfr(links_tbl$url, scrape_br_scouting_report)



get_player_links <- function(big_board_url) {
  read_html(big_board_url) |>
    html_elements('a[href*="scouting-report-for-"]') |>
    html_attr("href") |>
    unique() |>
    (\(x) ifelse(
      grepl("^https?://", x),
      x,
      paste0("https://bleacherreport.com", x)
    ))()
}


#2024
# big_board_url <- "https://bleacherreport.com/articles/10117937-br-nfl-scouting-depts-final-2024-nfl-draft-big-board"
# 
# player_links <- get_player_links(big_board_url)
# 
# future::plan(multisession, workers = 4)
# 
# all_reports <- future_map_dfr(player_links, scrape_br_scouting_report_faster)


#write.csv(all_reports, "Bleacher2024.csv")


# 2023
# big_board_url_2023 <- "https://bleacherreport.com/articles/10073286-2023-nfl-draft-big-board-br-nfl-scouting-depts-final-rankings"
# 
# player_links_2023 <- get_player_links(big_board_url_2023)
# 
# future::plan(multisession, workers = 4)
# 
# all_reports_2023 <- future_map_dfr(player_links_2023, scrape_br_scouting_report_faster)
# 
 # write.csv(all_reports_2023, "Bleacher2023.csv")


# ------------------
# 2022
# ------------------
# big_board_url_2022 <- "https://bleacherreport.com/articles/2955009-br-nfl-scouting-depts-final-2022-nfl-draft-big-board"
# 
# player_links_2022 <- get_player_links(big_board_url_2022)
# 
# all_reports_2022 <- future_map_dfr(
#   player_links_2022,
#   scrape_br_scouting_report_faster
# )
# 
# write.csv(all_reports_2022, "Bleacher2022.csv")


# ------------------
# 2021
# ------------------
# big_board_url_2021 <- "https://bleacherreport.com/articles/2932010-br-nfl-scouting-dept-final-2021-nfl-draft-big-board"
# 
# player_links_2021 <- get_player_links(big_board_url_2021)
# 
# all_reports_2021 <- future_map_dfr(
#   player_links_2021,
#   scrape_br_scouting_report_faster
# )
# 
# write.csv(all_reports_2021, "Bleacher2021.csv")


# ------------------
# 2025
# ------------------
big_board_url_2025 <- "https://bleacherreport.com/articles/25181593-2025-nfl-draft-big-board-br-nfl-scouting-depts-latest-rankings"

player_links_2025 <- get_player_links(big_board_url_2025)

all_reports_2025 <- future_map_dfr(
  player_links_2025,
  scrape_br_scouting_report_faster
)
#write.csv(all_reports_2025, "Bleacher2025.csv")

library(rvest)
library(stringr)
library(tibble)
library(dplyr)

scrape_br_2026_report <- function(article_url) {
  page <- read_html(article_url)
  
  clean1 <- function(x) {
    x <- x[1]
    if (length(x) == 0 || is.na(x) || identical(x, character(0))) return(NA_character_)
    str_squish(x)
  }
  
  safe_match <- function(text, pattern, group = 2) {
    out <- str_match(text, pattern)[, group]
    clean1(out)
  }
  
  section_between <- function(lines, start_pat, end_pat = NULL) {
    start_idx <- which(str_detect(lines, regex(start_pat, ignore_case = TRUE)))[1]
    
    if (is.na(start_idx)) return(NA_character_)
    
    if (is.null(end_pat)) {
      out <- lines[(start_idx + 1):length(lines)]
    } else {
      end_idx <- which(str_detect(lines, regex(end_pat, ignore_case = TRUE)))
      end_idx <- end_idx[end_idx > start_idx][1]
      
      if (is.na(end_idx)) {
        out <- lines[(start_idx + 1):length(lines)]
      } else {
        out <- lines[(start_idx + 1):(end_idx - 1)]
      }
    }
    
    out <- out[!is.na(out)]
    out <- str_squish(out)
    out <- out[out != ""]
    if (length(out) == 0) return(NA_character_)
    paste(out, collapse = " | ")
  }
  
  text_between <- function(lines, start_pat, end_pat = NULL) {
    start_idx <- which(str_detect(lines, regex(start_pat, ignore_case = TRUE)))[1]
    
    if (is.na(start_idx)) return(NA_character_)
    
    if (is.null(end_pat)) {
      out <- lines[(start_idx + 1):length(lines)]
    } else {
      end_idx <- which(str_detect(lines, regex(end_pat, ignore_case = TRUE)))
      end_idx <- end_idx[end_idx > start_idx][1]
      
      if (is.na(end_idx)) {
        out <- lines[(start_idx + 1):length(lines)]
      } else {
        out <- lines[(start_idx + 1):(end_idx - 1)]
      }
    }
    
    out <- out[!is.na(out)]
    out <- str_squish(out)
    out <- out[out != ""]
    if (length(out) == 0) return(NA_character_)
    str_squish(paste(out, collapse = " "))
  }
  
  # ---- metadata ----
  og_title <- page |>
    html_element('meta[property="og:title"]') |>
    html_attr("content") |>
    clean1()
  
  page_title <- page |>
    html_element("title") |>
    html_text2() |>
    clean1()
  
  title <- clean1(c(og_title, page_title))
  
  # ---- useful text nodes ----
  lines <- page |>
    html_elements("h1, h2, h3, h4, p, li, strong, b") |>
    html_text2() |>
    str_squish()
  
  lines <- lines[!is.na(lines)]
  lines <- lines[lines != ""]
  lines <- unique(lines)
  
  txt <- paste(lines, collapse = "\n")
  
  # ---- parse from URL as backup ----
  slug <- str_match(article_url, "articles/\\d+-([^/?#]+)")[, 2] |>
    clean1()
  
  slug_tail <- slug |>
    str_remove("^nfl-draft-\\d{4}-scouting-report-") |>
    str_replace_all("-", " ") |>
    clean1()
  
  # Try title first, then URL fallback
  player_from_title <- safe_match(
    title,
    "^(.+?)\\s+NFL Draft\\s+\\d{4}:"
  )
  
  if (is.na(player_from_title)) {
    # crude fallback for URLs like ohio-state-s-caleb-downs
    # takes final 2–4 words as player, leaving school/pos before it
    slug_parts <- str_split(slug_tail, "\\s+")[[1]]
    if (length(slug_parts) >= 2) {
      player_from_title <- paste(tail(slug_parts, 2), collapse = " ")
    }
  }
  
  # Better fallback from H1
  h1 <- page |>
    html_element("h1") |>
    html_text2() |>
    clean1()
  
  if (is.na(player_from_title) && !is.na(h1)) {
    player_from_title <- safe_match(h1, "^(.+?)\\s+NFL Draft")
  }
  
  # date
  date <- safe_match(
    txt,
    "((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.?\\s+\\d{1,2},\\s+\\d{4})",
    group = 1
  )
  
  # ---- measurements ----
  height   <- safe_match(txt, "HEIGHT:\\s*([^\\n]+?)(?:\\s+WEIGHT:|$)")
  weight   <- safe_match(txt, "WEIGHT:\\s*([^\\n]+?)(?:\\s+HAND:|\\s+ARM:|$)")
  hand     <- safe_match(txt, "HAND:\\s*([^\\n]+?)(?:\\s+ARM:|$)")
  arm      <- safe_match(txt, "ARM:\\s*([^\\n]+?)(?:\\s+WINGSPAN:|$)")
  wingspan <- safe_match(txt, "WINGSPAN:\\s*([^\\n]+?)(?:\\s+40-YARD DASH:|$)")
  dash40   <- safe_match(txt, "40-YARD DASH:\\s*([^\\n]+?)(?:\\s+3-CONE:|\\s+SHUTTLE:|$)")
  cone3    <- safe_match(txt, "3-CONE:\\s*([^\\n]+?)(?:\\s+SHUTTLE:|$)")
  shuttle  <- safe_match(txt, "SHUTTLE:\\s*([^\\n]+?)(?:\\s+VERTICAL:|$)")
  vertical <- safe_match(txt, "VERTICAL:\\s*([^\\n]+?)(?:\\s+BROAD:|$)")
  broad    <- safe_match(txt, "BROAD:\\s*([^\\n]+?)(?:\\s+POSITIVES|$)")
  
  # ---- sections ----
  positives <- section_between(
    lines,
    "^POSITIVES:?$",
    "^NEGATIVES:?$"
  )
  
  negatives <- section_between(
    lines,
    "^NEGATIVES:?$",
    "^(?:\\d{4}\\s+STATISTICS:?|STATISTICS:?|NOTES:?|OVERALL:?|GRADE:)"
  )
  
  statistics <- section_between(
    lines,
    "^(?:\\d{4}\\s+STATISTICS:?|STATISTICS:?)$",
    "^NOTES:?$|^OVERALL:?$|^GRADE:"
  )
  
  notes <- section_between(
    lines,
    "^NOTES:?$",
    "^OVERALL:?$|^GRADE:"
  )
  
  overall <- text_between(
    lines,
    "^OVERALL:?$",
    "^GRADE:"
  )
  
  article_grade <- safe_match(txt, "GRADE:\\s*([^\\n]+?)(?:\\s+OVERALL RANK:|$)")
  overall_rank  <- safe_match(txt, "OVERALL RANK:\\s*([^\\n]+?)(?:\\s+POSITION RANK:|$)")
  position_rank <- safe_match(txt, "POSITION RANK:\\s*([^\\n]+?)(?:\\s+PRO COMPARISON:|$)")
  pro_comparison <- safe_match(txt, "PRO COMPARISON:\\s*([^\\n]+?)(?:\\s+Written by|$)")
  
  tibble(
    url = article_url,
    title = title,
    h1 = h1,
    player_from_title = player_from_title,
    date = date,
    height = height,
    weight = weight,
    hand = hand,
    arm = arm,
    wingspan = wingspan,
    dash40 = dash40,
    cone3 = cone3,
    shuttle = shuttle,
    vertical = vertical,
    broad = broad,
    positives = positives,
    negatives = negatives,
    statistics = statistics,
    notes = notes,
    overall = overall,
    article_grade = article_grade,
    overall_rank = overall_rank,
    position_rank = position_rank,
    pro_comparison = pro_comparison,
    raw_slug = slug
  )
}


big_board_url_2026 <- "https://bleacherreport.com/articles/25384511-2026-nfl-draft-big-board-br-nfl-scouting-depts-post-senior-bowl-rankings"

# get_player_links_2026 <- function(big_board_url) {
#   read_html(big_board_url) |>
#     html_elements('a[href*="nfl-draft-2026-scouting-report"], a[href*="nfl-draft-2025-scouting-report"], a[href*="nfl-draft-2024-scouting-report"], a[href*="scouting-report-for-"]') |>
#     html_attr("href") |>
#     unique() |>
#     (\(x) x[!is.na(x)])() |>
#     (\(x) ifelse(
#       grepl("^https?://", x),
#       x,
#       paste0("https://bleacherreport.com", x)
#     ))() |>
#     unique()
# }

# test_2026 <- scrape_br_2026_report(
#   "https://bleacherreport.com/articles/25248468-nfl-draft-2026-scouting-report-ohio-state-s-caleb-downs"
# )
# 
# test_2026

player_links_2026 <- get_player_links_2026(big_board_url_2026)

all_reports_2026 <- future_map_dfr(
  player_links_2026,
  scrape_br_2026_report
)

v <- all_reports_2026

all_reports_2026_new <- all_reports_2026 |> 
  mutate(player_from_title = case_when(
    player_from_title == "Bain Jr." ~" Reuben Bain Jr."
    
  ))

extract_player_name <- function(title) {
  str_match(
    title,
    regex("\\b(QB|RB|WR|TE|OL|OT|OG|C|DL|EDGE|DT|DE|LB|CB|S)\\s+(.+)$", ignore_case = TRUE)
  )[, 3] |>
    str_squish()
}


all_reports_2026_new <- all_reports_2026 |>
  mutate(
    player_from_title = ifelse(is.na(player_from_title) | str_detect(position_rank, regex("edge", ignore_case = TRUE)), extract_player_name(title), player_from_title)
  ) |> 
  filter(!is.na(player_from_title))


#write.csv(all_reports_2026_new, "Bleacher2026.csv")

##### 2020
# 
# make_slug <- function(name) {
#   name |>
#     str_to_lower() |>
#     str_replace_all("[^a-z0-9\\s]", "") |>
#     str_replace_all("\\s+", "-")
# }
# 
# 
# get_2020_big_board_rows <- function(url) {
#   page <- read_html(url)
#   
#   lines <- page |>
#     html_element("body") |>
#     html_text2() |>
#     str_split("\n")
#   
#   lines <- lines[[1]] |>
#     str_squish()
#   
#   lines <- lines[lines != ""]
#   
#   tibble(raw = lines) |>
#     filter(
#       str_detect(raw, "^\\d+\\.\\s")
#     ) |>
#     mutate(
#       rank = str_extract(raw, "^\\d+") |> as.integer(),
#       inside = str_remove(raw, "^\\d+\\.\\s*"),
#       grade = str_match(inside, "\\(([^()]+)\\)$")[,2] |> str_squish(),
#       inside = str_remove(inside, "\\s*\\([^()]+\\)$"),
#       pieces = str_split(inside, ",\\s*")
#     ) |>
#     rowwise() |>
#     mutate(
#       n = length(pieces),
#       player = pieces[[1]],
#       pos_or_school_1 = ifelse(n >= 2, pieces[[2]], NA_character_),
#       pos_or_school_2 = ifelse(n >= 3, pieces[[3]], NA_character_)
#     ) |>
#     ungroup()
# }
# 
# 
# 
# get_2020_scouting_links <- function(big_board_url) {
#   
#   # --- helper: slug builder ---
#   make_slug <- function(name) {
#     name |>
#       stringr::str_to_lower() |>
#       stringr::str_replace_all("['’]", "") |>
#       stringr::str_replace_all("[^a-z0-9\\s]", "") |>
#       stringr::str_replace_all("\\s+", "-")
#   }
#   
#   # --- 1. get players ---
#   players_tbl <- get_2020_big_board_rows(big_board_url)
#   players <- players_tbl$player
#   
#   # --- 2. build slugs + patterns ---
#   slugs <- make_slug(players)
#   
#   patterns <- paste0(
#     slugs,
#     "-nfl-draft-2020-scouting-report"
#   )
#   
#   # --- 3. get all links on page ---
#   page <- rvest::read_html(big_board_url)
#   
#   all_links <- page |>
#     rvest::html_elements("a") |>
#     rvest::html_attr("href") |>
#     unique()
#   
#   all_links <- all_links[!is.na(all_links)]
#   
#   all_links <- ifelse(
#     stringr::str_detect(all_links, "^https?://"),
#     all_links,
#     paste0("https://bleacherreport.com", all_links)
#   )
#   
#   # --- 4. keep only scouting report URLs ---
#   scouting_links <- all_links[
#     stringr::str_detect(
#       all_links,
#       "bleacherreport\\.com/articles/\\d+-.+-nfl-draft-2020-scouting-report$"
#     )
#   ]
#   
#   # --- 5. match players to URLs ---
#   out <- tibble::tibble(
#     player = players,
#     slug = slugs,
#     pattern = patterns
#   ) |>
#     dplyr::rowwise() |>
#     dplyr::mutate(
#       url = scouting_links[
#         stringr::str_detect(scouting_links, pattern)
#       ][1]
#     ) |>
#     dplyr::ungroup()
#   
#   out
# }
# library(stringr)
# library(dplyr)
# library(tibble)
# library(purrr)
# library(httr2)
# library(rvest)
# library(xml2)
# 
# find_br_2020_scouting_url <- function(player_name, pause = 0.5) {
#   query <- paste0(
#     '"', player_name, '" ',
#     'site:bleacherreport.com/articles ',
#     '"nfl draft 2020" "scouting report"'
#   )
#   
#   search_url <- "https://html.duckduckgo.com/html/"
#   
#   resp <- request(search_url) |>
#     req_url_query(q = query) |>
#     req_user_agent("Mozilla/5.0") |>
#     req_perform()
#   
#   html <- resp |>
#     resp_body_html()
#   
#   hrefs <- html |>
#     html_elements("a") |>
#     html_attr("href")
#   
#   hrefs <- hrefs[!is.na(hrefs)]
#   
#   # keep only actual Bleacher Report article links
#   hrefs <- hrefs[
#     str_detect(
#       hrefs,
#       "bleacherreport\\.com/articles/\\d+-.+-nfl-draft-2020-scouting-report"
#     )
#   ]
#   
#   hrefs <- unique(hrefs)
#   
#   # small delay to be polite
#   Sys.sleep(pause)
#   
#   if (length(hrefs) == 0) return(NA_character_)
#   hrefs[1]
# }
# 
# 
# get_2020_scouting_links_via_search <- function(big_board_url, pause = 0.5) {
#   players_tbl <- get_2020_big_board_rows(big_board_url)
#   
#   players_tbl |>
#     distinct(player, .keep_all = TRUE) |>
#     mutate(
#       url = map_chr(player, find_br_2020_scouting_url, pause = pause)
#     )
# }
# 
# 
# big_board_url_2020 <- "https://bleacherreport.com/articles/2887115-matt-millers-final-2020-nfl-draft-big-board"
# 
# links_2020 <- get_2020_scouting_links_via_search(
#   big_board_url_2020,
#   pause = 0.75
# )
# 
# 
# players_tbl <- get_2020_big_board_rows(big_board_url_2020)
# 
# players <- players_tbl$player
# slugs <- make_slug(players)
# 
# prospect_links_2020 <- get_2020_scouting_links(big_board_url_2020)
# 
# prospect_links_2020
# 
