

#install.packages("pdftools")
library(pdftools)
library(stringr)
library(tibble)

# 2024!


txt <- pdf_text("~/Documents/STSCI 4981/NLP Project with Varun/PFF 2024 NFL Draft Guide.pdf")


pages <- tibble(
  page = seq_along(txt),
  text = txt
)


JackPlummer <- pages[34:37,]
cat(JackPlummer$text)


# Testing with Jack Plummer

name_block <- str_extract(JackPlummer |>
                            str_replace_all("•\\s*", "") |>
                            str_replace_all("\\\\n+", " ") |>
                            str_squish(), "^[\\s\\S]*?CLICK HERE for PFF Big Board") |>
  str_replace("^c\\(\"", "")
name_block[2]

first_name <- first_name <- str_match(name_block, "^([A-Za-z.]+)")[,2]

height <- str_match(name_block[2], "(\\d[’']\\d[\"”])")[,2]

last_name <- str_match(name_block[2], "([A-Za-z]+(?:\\s+(?:II|III|IV|Jr\\.?|Sr\\.?))?)\\s+\\d{3}\\s*lbs")[,2]

weight <- str_match(name_block[2], "(\\d{3})\\s*lbs")[,2]

position <- str_match(name_block[2], "(Quarterback|Running Back|Wide Receiver|Tight End|Linebacker|Cornerback|Safety|Offensive Line|Edge Defender|Interior Defender)")[,2]


profile <- str_match(JackPlummer |>
                       str_replace_all("\\n+", "\n") |>
                       str_squish(), "PROFILE\\s*([\\s\\S]*?)PLAYER TRAITS")[,2]
profile[2]

strengths <- str_match(JackPlummer |>
                         str_replace_all("•\\s*", "") |>
                         str_replace_all("\\\\n+", " ") |>
                         str_squish(),
                       "STRENGTHS\\s*([\\s\\S]*?)WEAKNESSES")[,2]

strengths[2]

weaknesses <- str_match(JackPlummer |>
                          str_replace_all("•\\s*", "") |>
                          str_replace_all("\\\\n+", " ") |>
                          str_squish(), "WEAKNESSES\\s*([\\s\\S]*?)PLAYER COMP")[,2]
weaknesses[2]

player <- str_match(JackPlummer |>
                           str_replace_all("\\\\n+", "") |>
                           str_replace_all("\"", "") |>
                           str_replace_all(",", "") |>
                           str_squish(), "PLAYER COMP\\s*([\\s\\S]*?)PFF GRADES BY FACET")[,2] |>
  str_trim()
player[2]

bottom_line <- str_match(
  JackPlummer$text |>
    str_replace_all("\\n+", "\n"),
  "BOTTOM line\\s*([\\s\\S]*)"
)[,2] |>
  str_squish()
bottom_line[4]



#### The Big generalization, in function form

parse_prospect_2024 <- function(JackPlummer) {
  
  name_block <- str_extract(
    JackPlummer |>
      str_replace_all("•\\s*", "") |>
      str_replace_all("\\\\n+", " ") |>
      str_squish(),
    "^[\\s\\S]*?CLICK HERE for PFF Big Board"
  ) |> str_replace("^c\\(\"", "") |> 
    str_replace_all("’", "'") |>
    str_replace_all("`", "'")
  
  first_name <- str_match(name_block, "^([A-Za-z][A-Za-z.'-]*)")[,2]
  height <- str_match(name_block, "(\\d[’']\\d[\"”])")[,2]
  last_name <- str_match(name_block, "([A-Za-z]+(?:\\s+(?:II|III|IV|Jr\\.?|Sr\\.?))?)\\s+\\d{3}\\s*lbs")[,2]
  weight <- str_match(name_block, "(\\d{3})\\s*lbs")[,2]
  position <- str_match(
    name_block,
    "(Quarterback|Running Back|Wide Receiver|Tight End|Linebacker|Cornerback|Safety|Offensive Line|Edge Defender|Interior Defender|Box Safety|Free Safety|Tackle|Interior OL)"
  )[,2]
  
  profile <- str_match(
    JackPlummer |>
      str_replace_all("•\\s*", "") |>
      str_replace_all("\\\\n+", " ") |>
      str_squish(),
    "PROFILE\\s*([\\s\\S]*?)PLAYER TRAITS"
  )[,2]
  
  strengths <- str_match(
    JackPlummer |>
      str_replace_all("•\\s*", "") |>
      str_replace_all("\\\\n+", " ") |>
      str_squish(),
    "STRENGTHS\\s*([\\s\\S]*?)WEAKNESSES"
  )[,2]
  
  weaknesses <- str_match(
    JackPlummer |>
      str_replace_all("•\\s*", "") |>
      str_replace_all("\\\\n+", " ") |>
      str_squish(),
    "WEAKNESSES\\s*([\\s\\S]*?)PLAYER COMP"
  )[,2]
  
  player <- str_match(
    JackPlummer |>
      str_replace_all("\\\\n+", "") |>
      str_replace_all("\"", "") |>
      str_replace_all(",", "") |>
      str_squish(),
    "PLAYER COMP\\s*([\\s\\S]*?)PFF GRADES BY FACET"
  )[,2] |>
    str_trim()
  
  bottom_line <- str_match(
    JackPlummer |>
      str_replace_all("•\\s*", "") |>
      str_replace_all("\\\\n+", " ") |>
      str_squish(),
    "BOTTOM line\\s*([\\s\\S]*)"
  )[,2] |>
    str_squish()
  
  tibble(
    first_name = first_name,
    last_name = last_name,
    height = height,
    weight = weight,
    position = position,
    profile = profile,
    strengths = strengths,
    weaknesses = weaknesses,
    player = player,
    bottom_line = bottom_line
  )
}

start_pages <- pages[
  grepl("CLICK HERE for PFF Big Board", pages$text) &
    grepl("PROFILE", pages$text) &
    grepl("PLAYER TRAITS", pages$text),
]

prospect_blocks <- lapply(start_pages$page, function(i) {
  paste(pages$text[i:min(i + 3, nrow(pages))], collapse = " ")
})

all_prospects <- lapply(prospect_blocks, parse_prospect_2024)
table_2024 <- do.call(rbind, all_prospects)

# DEAL WITH TVONDRE SWEAT LATER

#write.csv(table_2024, "2024_data.csv")


########## 2022!



txt <- pdf_text("~/Documents/STSCI 4981/NLP Project with Varun/PFF 2022 NFL Draft Guide.pdf")


pages <- tibble(
  page = seq_along(txt),
  text = txt
)


# JackPlummer <- pages[13:15,]
# cat(JackPlummer$text)
# 
# 
# block <- paste(JackPlummer$text, collapse = "\n") |>
#   str_replace_all("\\n+", "\n")
# 
# cat(block, sep = "\n")
# 
# 
# # First name
# full_name <- block |>
#   str_extract("\\n\\s*[A-Z][a-z]+\\s*\\n\\s*[A-Z]{2,}") |>
#   str_replace_all("\\n", " ") |>
#   str_squish() |>
#   str_to_title()
# 
# 
# # BLURB
# blurb <- str_match(
#   block,
#   "- Via 247 Sports\\s*([\\s\\S]*?)\\s*SEASON STATS"
# )[,2]
# 
# lines <- str_split(blurb, "\n")[[1]]
# 
# 
# lines_clean <- str_squish(lines)
# 
# lines_clean <- lines_clean |>
#   str_replace_all(
#     "POSITION RK\\.|BIG BOARD RK\\.|ROUND PROJ\\.|SHADES OF|Hometown|Long Beach, CA",
#     ""
#   ) |>
#   str_squish()
# 
# lines_clean <- lines_clean |>
#   str_replace_all("\\s+\\d+(st|nd|rd|th)?$", "")
# 
# lines_clean <- lines_clean[
#   str_count(lines_clean, "\\S+") > 1 &   # more than 1 word
#     !str_detect(lines_clean, "^\\d+$")     # not just numbers
# ]
# 
# blurb <- paste(lines_clean, collapse = " ")
# 
# 
# # PROS AND CONS
# 
# pc_block <- str_extract(
#   block,
#   "P\\s*R\\s*O\\s*S[\\s\\S]*?Return to Table of Contents"
# )
# 
# lines <- str_split(pc_block, "\n")[[1]]
# 
# split_lines <- str_split(lines, "\\s{5,}")
# 
# 
# split_lines <- lapply(split_lines, function(row) {
#   length(row) <- 3
#   
#   if (str_squish(row[1]) == "") {
#     # shift everything left
#     row[1] <- row[2]
#     row[2] <- row[3]
#     row[3] <- ""
#   }
#   
#   row
# })
# 
# pros_before <- sapply(split_lines, function(row) str_squish(row[1]))
# cons_before <- sapply(split_lines, function(row) str_squish(row[2]))
# 
# 
# clean_vec <- function(x, symbol_pattern) {
#   
#   x <- str_squish(x)
#   
#   # remove headers / footer
#   keep <- !str_detect(x, "P R OS|CON S|Return to Table of Contents")
#   x <- x[keep]
#   
#   # remove NA / empty
#   x <- x[!is.na(x) & x != ""]
#   
#   # remove symbols
#   x <- str_remove(x, paste0("^", symbol_pattern, "\\s*"))
#   
#   # merge continuation lines
#   out <- c()
#   current <- ""
#   
#   for (line in x) {
#     if (str_detect(line, "^[A-Z]")) {
#       if (current != "") out <- c(out, current)
#       current <- line
#     } else {
#       current <- paste(current, line)
#     }
#   }
#   
#   if (current != "") out <- c(out, current)
#   
#   str_squish(out)
# }
# 
# 
# pros_clean <- clean_vec(pros_before, "Ø")
# cons_clean <- clean_vec(cons_before, "v")
# 
# pros <- paste(pros_clean, collapse = " ")
# cons <- paste(cons_clean, collapse = " ")
# 
# 
# 
# 
# ## THE OTHER STUFF
# 
# get_section <- function(text, start, end) {
#   str_match(text, paste0(start, "\\s*([\\s\\S]*?)\\s*", end))[,2] |>
#     str_squish()
# }
# 
# where_wins <- get_section(block, "WINS", "W h")
# 
# what_role <- get_section(block, "ROLE", "W h")
# 
# what_improve <- get_section(block, "IMPROVE", "Return")
# 
# # Bottom Line
# 
# bottom_line <- get_section(block, "THE BOTTOM LINE", "Return")
# 



###################### FUNCTION

parse_prospect_2022 <- function(pages_subset) {
  
  # --- BLOCK ---
  block <- paste(pages_subset, collapse = "\n")
  block <- str_replace_all(block, "\\n+", "\n")
  
  # --- NAME ---
  full_name <- block |>
    str_extract("\\n\\s*[A-Z][a-z]+\\s*\\n\\s*[A-Z]{2,}") |>
    str_replace_all("\\n", " ") |>
    str_squish() |>
    str_to_title()
  
  # --- BLURB ---
  blurb_raw <- str_match(
    block,
    "- Via 247 Sports\\s*([\\s\\S]*?)\\s*SEASON STATS"
  )[,2]
  
  lines <- str_split(blurb_raw, "\n")[[1]]
  
  lines_clean <- str_squish(lines)
  
  lines_clean <- str_replace_all(
    lines_clean,
    "POSITION RK\\.|BIG BOARD RK\\.|ROUND PROJ\\.|SHADES OF|Hometown|Long Beach, CA",
    ""
  )
  
  lines_clean <- str_squish(lines_clean)
  
  lines_clean <- str_replace_all(lines_clean, "\\s+\\d+(st|nd|rd|th)?$", "")
  
  lines_clean <- lines_clean[
    str_count(lines_clean, "\\S+") > 1 &
      !str_detect(lines_clean, "^\\d+$")
  ]
  
  blurb <- paste(lines_clean, collapse = " ")
  
  # --- PROS / CONS ---
  pc_block <- str_extract(
    block,
    "P\\s*R\\s*O\\s*S[\\s\\S]*?Return to Table of Contents"
  )
  
  lines <- str_split(pc_block, "\n")[[1]]
  split_lines <- str_split(lines, "\\s{5,}")
  
  split_lines <- lapply(split_lines, function(row) {
    length(row) <- 3
    
    if (str_squish(row[1]) == "") {
      row[1] <- row[2]
      row[2] <- row[3]
      row[3] <- ""
    }
    
    row
  })
  
  pros_before <- sapply(split_lines, function(row) str_squish(row[1]))
  cons_before <- sapply(split_lines, function(row) str_squish(row[2]))
  
  clean_vec <- function(x, symbol_pattern) {
    
    x <- str_squish(x)
    
    keep <- !str_detect(x, "P R OS|CON S|Return to Table of Contents")
    x <- x[keep]
    
    x <- x[!is.na(x) & x != ""]
    
    x <- str_remove(x, paste0("^", symbol_pattern, "\\s*"))
    
    out <- c()
    current <- ""
    
    for (line in x) {
      if (str_detect(line, "^[A-Z]")) {
        if (current != "") out <- c(out, current)
        current <- line
      } else {
        current <- paste(current, line)
      }
    }
    
    if (current != "") out <- c(out, current)
    
    str_squish(out)
  }
  
  pros_clean <- clean_vec(pros_before, "Ø")
  cons_clean <- clean_vec(cons_before, "v")
  
  pros <- paste(pros_clean, collapse = " ")
  cons <- paste(cons_clean, collapse = " ")
  
  # --- OTHER SECTIONS ---
  get_section <- function(text, start, end) {
    str_match(text, paste0(start, "\\s*([\\s\\S]*?)\\s*", end))[,2] |>
      str_squish()
  }
  
  where_wins <- get_section(block, "WINS", "W h")
  what_role <- get_section(block, "ROLE", "W h")
  what_improve <- get_section(block, "IMPROVE", "Return")
  
  bottom_line <- get_section(block, "THE BOTTOM LINE", "Return")
  
  # --- OUTPUT ---
  tibble::tibble(
    full_name = full_name,
    blurb = blurb,
    pros = pros,
    cons = cons,
    where_wins = where_wins,
    what_role = what_role,
    what_improve = what_improve,
    bottom_line = bottom_line
  )
}


start_pages <- pages[
  grepl("Via 247 Sports", pages$text) &
    grepl("POSITION RK\\.", pages$text),
]

start_idxs <- start_pages$page
n <- length(start_idxs)

prospect_blocks <- lapply(seq_along(start_idxs), function(k) {
  
  start <- start_idxs[k]
  
  end <- if (k < n) {
    start_idxs[k + 1] - 1
  } else {
    nrow(pages)
  }
  
  paste(pages$text[start:end], collapse = " ")
})


all_prospects <- lapply(prospect_blocks, parse_prospect_2022)
table_2022 <- do.call(rbind, all_prospects)

#write.csv(table_2022, "2022_data.csv")



# 2021 -- DONE NAME, SCHOOL, INITIAL SCOUTING REPORT

txt <- pdf_text("~/Documents/STSCI 4981/NLP Project with Varun/PFF 2021 NFL Draft Guide.pdf")


pages <- tibble(
  page = seq_along(txt),
  text = txt
)

# JackPlummer <- pages[94:97,]
# cat(JackPlummer$text)
# 
# 
# block <- paste(JackPlummer$text, collapse = "\n") |>
#   str_replace_all("\\n+", "\n")
# 
# cat(block, sep = "\n")
# # 
# blurb <- str_match(
#   block,
#   "\\b(?:OT|S)\\b\\s*([\\s\\S]*?)\\s*GAME-BY-GAME GRADES"
# )[,2]
# 
# lines <- str_split(blurb, "\n")[[1]]
# 
# segments <- str_split(lines, "\\s{10,}")
# 
# 
# blurb_parts <- unlist(segments)
# 
# blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
# 
# first_name <- blurb_parts[3] |> str_squish()
# last_name <- blurb_parts[5] |> str_squish()
# 
# name <- paste(first_name, last_name) |>
#   str_squish() |>
#   str_to_title()
# 
# 
# blurb_parts <- blurb_parts[-c(3, 5)]
# 
# blurb_parts <- blurb_parts[
#   !str_detect(blurb_parts, "^School$|^\\d{4}$|^3$|^5$")
# ]
# 
# # School
# 
# school <- tail(blurb_parts, 1) |> str_squish()
# 
# blurb_parts <- head(blurb_parts, -1)
# 
# 
# # Blurb
# 
# blurb <- paste(blurb_parts, collapse = " ") |>
#   str_squish()
# 
# 
# 
# #### Second blurb
# 
# blurb2 <- str_match(
#   block,
#   "\\Table of Contents\\b\\s*([\\s\\S]*?)\\s*PROS"
# )[,2]
# 
# lines <- str_split(blurb2, "\n")[[1]]
# 
# segments <- str_split(lines, "\\s{10,}")
# 
# blurb_parts <- unlist(segments)
# 
# blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
# 
# blurb_parts <- blurb_parts[-c(1, 2, 9)]
# 
# blurb_parts <- blurb_parts[
#   !str_detect(blurb_parts, "^School$|^\\d{4}$|^3$|^5$")
# ]
# 
# blurb2 <- paste(blurb_parts, collapse = " ") |>
#   str_squish()
# 
# 
# ### PROS AND CONS
# 
# pc_block <- str_match(
#   block,
#   "PROS AND CONS\\s*([\\s\\S]*?)\\s*NFL DRAFT"
# )[,2]
# 
# lines <- str_split(pc_block, "\n")[[1]]
# 
# parts <- unlist(lines)
# parts <- str_squish(parts)
# parts <- parts[!is.na(parts) & parts != ""]
# 
# parts <- parts[!str_detect(
#   parts,
#   "STAT COMPARABLES|BAR HEIGHT DENOTES PERCENTILE|AVG\\.?|^\\d+(\\.\\d+)?$|^GRADE$"
# )]
# 
# parts <- parts[str_detect(parts, "[A-Za-z]")]
# 
# parts_clean <- parts |>
#   str_replace_all("\\b[A-Z]{2,}(?:\\s+[A-Z]{2,})*\\b", "") |>
#   str_replace_all("\\b\\d+(\\.\\d+)?%?\\b", "") |>
#   str_squish() |> 
#   na.omit()
# 
# # Has the bottom line blurb, but NLP should be able to detect this anyway
# pros_a_cons <- paste(parts_clean, collapse = " ") |>
#   str_squish()




#### Function for 2021


parse_prospect_2021 <- function(block) {
  
  block <- paste(block, collapse = "\n") |>
    str_replace_all("\\n+", "\n")
  
  # --- First blurb / name / school ---
  blurb1 <- str_match(
    block,
    "\\b(?:OT|S)\\b\\s*([\\s\\S]*?)\\s*GAME-BY-GAME GRADES"
  )[,2]
  
  lines <- str_split(blurb1, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
  
  first_name <- blurb_parts[3] |> str_squish()
  last_name <- blurb_parts[5] |> str_squish()
  
  name <- paste(first_name, last_name) |>
    str_squish() |>
    str_to_title()
  
  blurb_parts <- blurb_parts[-c(3, 5)]
  blurb_parts <- blurb_parts[
    !str_detect(blurb_parts, "^School$|^\\d{4}$|^3$|^5$")
  ]
  
  school <- tail(blurb_parts, 1) |>
    str_squish()
  
  blurb_parts <- head(blurb_parts, -1)
  
  blurb1 <- paste(blurb_parts, collapse = " ") |>
    str_squish()
  
  # --- Second blurb ---
  blurb2 <- str_match(
    block,
    "Table of Contents\\s*([\\s\\S]*?)\\s*PROS"
  )[,2]
  
  lines <- str_split(blurb2, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
  blurb_parts <- blurb_parts[-c(1, 2, 9)]
  blurb_parts <- blurb_parts[
    !str_detect(blurb_parts, "^School$|^\\d{4}$|^3$|^5$")
  ]
  
  blurb2 <- paste(blurb_parts, collapse = " ") |>
    str_squish()
  
  # --- Pros and cons block ---
  pc_block <- str_match(
    block,
    "PROS AND CONS\\s*([\\s\\S]*?)\\s*NFL DRAFT"
  )[,2]
  
  lines <- str_split(pc_block, "\n")[[1]]
  
  parts <- unlist(lines)
  parts <- str_squish(parts)
  parts <- parts[!is.na(parts) & parts != ""]
  
  parts <- parts[!str_detect(
    parts,
    "STAT COMPARABLES|BAR HEIGHT DENOTES PERCENTILE|AVG\\.?|^\\d+(\\.\\d+)?$|^GRADE$"
  )]
  
  parts <- parts[str_detect(parts, "[A-Za-z]")]
  
  parts_clean <- parts |>
    str_replace_all("\\b[A-Z]{2,}(?:\\s+[A-Z]{2,})*\\b", "") |>
    str_replace_all("\\b\\d+(\\.\\d+)?%?\\b", "") |>
    str_squish() |>
    na.omit()
  
  pros_a_cons <- paste(parts_clean, collapse = " ") |>
    str_squish()
  
  tibble::tibble(
    full_name = name,
    school = school,
    blurb1 = blurb1,
    blurb2 = blurb2,
    pros_a_cons = pros_a_cons
  )
}

start_pages <- pages[
  grepl("247 Sports", pages$text) &
    grepl("Class", pages$text),
]

start_idxs <- start_pages$page
n <- length(start_idxs)

prospect_blocks <- lapply(seq_along(start_idxs), function(k) {
  
  start <- start_idxs[k]
  
  end <- if (k < n) {
    start_idxs[k + 1] - 1
  } else {
    nrow(pages)
  }
  
  paste(pages$text[start:end], collapse = " ")
})


all_prospects <- lapply(prospect_blocks, parse_prospect_2021)
table_2021 <- do.call(rbind, all_prospects)

#write.csv(table_2021, "2021_data.csv")


# Can go back and manually clean the names


### do 2020

txt <- pdf_text("~/Documents/STSCI 4981/NLP Project with Varun/PFF 2020 NFL Draft Guide.pdf")


pages <- tibble(
  page = seq_along(txt),
  text = txt
)

# JackPlummer <- pages[13:15,]
# cat(JackPlummer$text)
#
#
 # block <- paste(JackPlummer$text, collapse = "\n") |>
 #   str_replace_all("\\n+", "\n")
#
# cat(block, sep = "\n")


# # --- First blurb / name / school ---
# blurb1 <- str_match(
#   block,
#   "\\bNFL DRAFT GUIDE\\b\\s*([\\s\\S]*?)\\s*\\bROUND\\b"
# )[,2]
#
# lines <- str_split(blurb1, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
#
# blurb_parts <- unlist(segments)
# blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
#
# name <- blurb_parts[3] |>
#   str_squish() |>
#   str_to_title()
#
# position <- blurb_parts[10] |>
#   str_squish() |>
#   str_to_title()
#
# blurb_clean <- blurb_parts[str_detect(blurb_parts, "[a-z]")]
#
# blurb_clean <- blurb_clean |>
#   str_replace_all("\\s{2,}.*$", "") |>
#   str_squish()
#
# blurb1_final <- paste(blurb_clean, collapse = " ") |>
#   str_squish()
#
#
#
# # --- Second Blurb ---
# blurb2 <- str_match(
#   block,
#   "\\bLINE\\b\\s*([\\s\\S]*?)\\s*\\bNFL PLAYER COMPARISON\\b"
# )[,2]
#
# lines <- str_split(blurb1, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
#
# blurb_parts <- unlist(segments)
# blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
#
# blurb_clean <- blurb_parts[str_detect(blurb_parts, "[a-z]")]
#
# blurb2_prob <- blurb_parts[str_detect(blurb_parts, "^\\s+")]
# blurb2_prob <- blurb2_prob[
#   !str_detect(
#     blurb2_prob,
#     "^\\s*\\d{4}$|NFL DRAFT GUIDE|MIKE RENNER’S"
#   )
# ]
# blurb2_prob_clean <- str_split(blurb2_prob, "\\s{2,}")
#
# result <- lapply(blurb2_prob_clean, function(x) {
#   x <- str_squish(x)
#   x <- x[x != ""]
#
#   if (length(x) == 0) {
#     return(list(main = character(0), extra = character(0)))
#   }
#
#   list(
#     main  = x[1],
#     extra = if (length(x) > 1) x[-1] else character(0)
#   )
# })
#
# blurb2_prob_clean <- lapply(result, function(r) r$main)
#
# blurb2_prob_clean <- unlist(blurb2_prob_clean, use.names = FALSE)
# blurb2_prob_clean <- blurb2_prob_clean[blurb2_prob_clean != "" & !is.na(blurb2_prob_clean)]
#
# blurb2_prob_clean_FINAL<- paste(blurb2_prob_clean, collapse = " ") |>
#   str_squish()
#
# blurb2_extra      <- lapply(result, function(r) r$extra)
# blurb2_extra <- unlist(blurb2_extra, use.names = FALSE)
# blurb2_extra <- blurb2_extra[blurb2_extra != "" & !is.na(blurb2_extra)]
#
# # Pros and cons
#
# blurb2_not <- blurb_parts[!str_detect(blurb_parts, "^\\s+")]
#
# blurb2_extra_2 <- c(blurb2_extra, blurb2_not)
#
# blurb2_extra_2_clean <- blurb2_extra_2[str_detect(blurb2_extra_2, "[a-z]")]
#
# blurb2_extra_2_clean <- blurb2_extra_2_clean[
#   !str_detect(blurb2_extra_2_clean, regex("Table of Contents", ignore_case = TRUE))
# ]
#
# blurb2_extra_2_clean_final <- paste(blurb2_extra_2_clean, collapse = " ") |>
#   str_squish()



parse_prospect_2020 <- function(JackPlummer) {
  
  #JackPlummer <- pages[13:15,]
  
  block <- paste(JackPlummer, collapse = "\n") |>
    str_replace_all("\\n+", "\n")
  
  # --- First blurb / name / school ---
  blurb1 <- str_match(
    block,
    "\\bNFL DRAFT GUIDE\\b\\s*([\\s\\S]*?)\\s*\\bROUND\\b"
  )[,2]
  
  lines <- str_split(blurb1, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
  
  name <- blurb_parts[
    str_detect(
      blurb_parts,
      "^[A-Z][A-Z\\.\\-']*(?:\\s+[A-Z][A-Z\\.\\-']*)*(?:\\s+(?:JR\\.?|SR\\.?|II|III|IV))?$"
    ) &
      !str_detect(blurb_parts, "QUICK FACTS|QUARTERBACK|POSITION|RANK|GUIDE")
  ][1] |>
    str_squish() |>
    str_to_title()
  
  position <- blurb_parts[
    str_detect(blurb_parts, "^[A-Z]{2,}(?:\\s+[A-Z]{2,})*$")
  ][2] |>
    str_squish() |>
    str_to_title()
  
  blurb_clean <- blurb_parts[str_detect(blurb_parts, "[a-z]")]
  
  blurb_clean <- blurb_clean |>
    str_replace_all("\\s{2,}.*$", "") |>
    str_squish()
  
  blurb1_final <- paste(blurb_clean, collapse = " ") |>
    str_squish()
  
  # --- Second blurb ---
  blurb2 <- str_match(
    block,
    "\\bLINE\\b\\s*([\\s\\S]*?)\\s*\\bNFL PLAYER COMPARISON\\b"
  )[,2]
  
  lines <- str_split(blurb2, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[!is.na(blurb_parts) & blurb_parts != ""]
  
  blurb_clean <- blurb_parts[str_detect(blurb_parts, "[a-z]")]
  
  blurb2_prob <- blurb_parts[str_detect(blurb_parts, "^\\s+")]
  blurb2_prob <- blurb2_prob[
    !str_detect(
      blurb2_prob,
      "^\\s*\\d{4}$|NFL DRAFT GUIDE|MIKE RENNER’S"
    )
  ]
  
  blurb2_prob_clean <- str_split(blurb2_prob, "\\s{2,}")
  
  result <- lapply(blurb2_prob_clean, function(x) {
    x <- str_squish(x)
    x <- x[x != ""]
    
    if (length(x) == 0) {
      return(list(main = character(0), extra = character(0)))
    }
    
    list(
      main  = x[1],
      extra = if (length(x) > 1) x[-1] else character(0)
    )
  })
  
  blurb2_prob_clean <- lapply(result, function(r) r$main)
  
  blurb2_prob_clean <- unlist(blurb2_prob_clean, use.names = FALSE)
  blurb2_prob_clean <- blurb2_prob_clean[
    blurb2_prob_clean != "" & !is.na(blurb2_prob_clean)
  ]
  
  blurb2_prob_clean_FINAL <- paste(blurb2_prob_clean, collapse = " ") |>
    str_squish()
  
  blurb2_extra <- lapply(result, function(r) r$extra)
  blurb2_extra <- unlist(blurb2_extra, use.names = FALSE)
  blurb2_extra <- blurb2_extra[blurb2_extra != "" & !is.na(blurb2_extra)]
  
  # Pros and cons
  blurb2_not <- blurb_parts[!str_detect(blurb_parts, "^\\s+")]
  
  blurb2_extra_2 <- c(blurb2_extra, blurb2_not)
  
  blurb2_extra_2_clean <- blurb2_extra_2[str_detect(blurb2_extra_2, "[a-z]")]
  
  blurb2_extra_2_clean <- blurb2_extra_2_clean[
    !str_detect(blurb2_extra_2_clean, regex("Table of Contents", ignore_case = TRUE))
  ]
  
  blurb2_extra_2_clean_final <- paste(blurb2_extra_2_clean, collapse = " ") |>
    str_squish()
  
  tibble(
    name = name,
    position = position,
    blurb1 = blurb1_final,
    blurb2 = blurb2_prob_clean_FINAL,
    pros_and_cons = blurb2_extra_2_clean_final
  )
}




start_pages <- pages[
  grepl("QUICK FACTS", pages$text) &
    grepl("BIG-BOARD", pages$text),
]

start_idxs <- start_pages$page
n <- length(start_idxs)

prospect_blocks <- lapply(seq_along(start_idxs), function(k) {
  
  start <- start_idxs[k]
  
  end <- if (k < n) {
    start_idxs[k + 1] - 1
  } else {
    nrow(pages)
  }
  
  paste(pages$text[start:end], collapse = " ")
})


all_prospects <- lapply(prospect_blocks, parse_prospect_2020)
table_2020 <- do.call(rbind, all_prospects)

#write.csv(table_2020, "2020_data.csv")



txt <- pdf_text("~/Documents/STSCI 4981/NLP Project with Varun/PFF 2018 NFL Draft Guide.pdf")


pages <- tibble(
  page = seq_along(txt),
  text = txt
)


# JackPlummer <- pages[15,]
# cat(JackPlummer$text)
# 
# 
# block <- paste(JackPlummer$text, collapse = "\n") |>
#   str_replace_all("\\n+", "\n")
# 
# cat(block, sep = "\n")
# 
# name_block <- str_match(
#   block,
#   "\\b2018 NFL DRAFT GUIDE\\s*([\\s\\S]*?)\\s*CL:"
# )[,2]
# 
# lines <- str_split(name_block, "\n")[[1]]
# 
# position <- str_extract(lines[1], "^[A-Z]{1,3}")
# name_raw <- str_remove(lines[1], "^[A-Z]{1,3}\\s+") |>
#   str_to_title()
# 
# college <- str_remove(lines[2], "^TEAM:\\s*") |>
#   str_squish() |>
#   str_to_title()
# 
# 
# blurb1 <- str_match(
#   block,
#   "\\bOVERVIEW:\\s*([\\s\\S]*?)\\s*BOTTOM\\s+LINE:"
# )[,2]
# 
# lines <- str_split(blurb1, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
# 
# blurb_parts <- unlist(segments)
# 
# blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
#   str_replace_all("•", "") |>  
#   str_squish() 
# 
# blurb_final <- paste(blurb_parts, collapse = " ") |>
#   str_squish()
# 
# # The bottom line
# 
# blurb2 <- str_match(
#   block,
#   "\\bBOTTOM\\s+LINE:\\s*([\\s\\S]*?)\\s*©"
# )[,2]
# 
# lines <- str_split(blurb2, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
# 
# blurb_parts <- unlist(segments)
# 
# blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
#   str_replace_all("•", "") |>  
#   str_squish() 
# 
# blurb2_final <- paste(blurb_parts, collapse = " ") |>
#   str_squish()



parse_prospect_2018 <- function(JackPlummer) {
  
  block <- paste(JackPlummer, collapse = "\n") |>
    str_replace_all("\\n+", "\n")
  
  # --- Header block: position, name, college ---
  name_block <- str_match(
    block,
    "\\b2018 NFL DRAFT GUIDE\\s*([\\s\\S]*?)\\s*CL:"
  )[,2]
  
  lines <- str_split(name_block, "\n")[[1]]
  lines <- str_squish(lines)
  lines <- lines[lines != ""]
  
  position <- str_extract(lines[1], "^[A-Z]{1,3}")
  
  name_raw <- str_remove(lines[1], "^[A-Z]{1,3}\\s+") |>
    str_squish() |>
    str_to_title()
  
  college <- str_remove(lines[2], "^TEAM:\\s*") |>
    str_squish() |>
    str_to_title()
  
  # --- Overview ---
  blurb1 <- str_match(
    block,
    "\\bOVERVIEW:\\s*([\\s\\S]*?)\\s*BOTTOM\\s+LINE:"
  )[,2]
  
  lines <- str_split(blurb1, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
    str_replace_all("•", "") |>
    str_squish()
  
  blurb_final <- paste(blurb_parts, collapse = " ") |>
    str_squish()
  
  # --- Bottom line ---
  blurb2 <- str_match(
    block,
    "\\bBOTTOM\\s+LINE:\\s*([\\s\\S]*?)\\s*©"
  )[,2]
  
  lines <- str_split(blurb2, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
    str_replace_all("•", "") |>
    str_squish()
  
  blurb2_final <- paste(blurb_parts, collapse = " ") |>
    str_squish()
  
  tibble(
    position = position,
    name = name_raw,
    college = college,
    overview = blurb_final,
    bottom_line = blurb2_final
  )
}


start_pages <- pages[
  grepl("OVERVIEW:", pages$text) &
    grepl("BOTTOM\\s+LINE:", pages$text),
]

start_idxs <- start_pages$page
n <- length(start_idxs)

prospect_blocks <- lapply(seq_along(start_idxs), function(k) {
  
  start <- start_idxs[k]
  
  end <- if (k < n) {
    start_idxs[k + 1] - 1
  } else {
    nrow(pages)
  }
  
  paste(pages$text[start:end], collapse = " ")
})


all_prospects <- lapply(prospect_blocks, parse_prospect_2018)
table_2018 <- do.call(rbind, all_prospects)

#write.csv(table_2018, "2018_data.csv")


txt <- pdf_text("~/Documents/STSCI 4981/NLP Project with Varun/PFF 2017 NFL Draft Guide.pdf")


pages <- tibble(
  page = seq_along(txt),
  text = txt
)


JackPlummer <- pages[210,]
cat(JackPlummer$text)

# 
# block <- paste(JackPlummer$text, collapse = "\n") |>
#   str_replace_all("\\n+", "\n")
# 
 cat(block, sep = "\n")
# 
# # Pros
# blurb1 <- str_match(
#   block,
#   "\\bWhat he does best:\\s*([\\s\\S]*?)\\s*Biggest concerns:"
# )[,2]
# 
# lines <- str_split(blurb1, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
# 
# blurb_parts <- unlist(segments)
# blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
#   str_squish()
# 
# blurb_parts <- blurb_parts[
#   str_detect(blurb_parts, "^\\s*•") | !str_detect(blurb_parts, "^\\s*[A-Z]")
# ]
# 
# pros <- paste(blurb_parts, collapse = " ") |>
#   str_replace_all("•", "") |>
#   str_squish()
# 
# # Cons
# cons1 <- str_match(
#   block,
#   "\\bBiggest concerns:\\s*([\\s\\S]*?)\\s*Bottom line:"
# )[,2]
# 
# lines <- str_split(cons1, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
# 
# blurb_parts <- unlist(segments)
# blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
#   str_squish()
# 
# blurb_parts <- blurb_parts[
#   str_detect(blurb_parts, "^\\s*•") | !str_detect(blurb_parts, "^\\s*[A-Z]")
# ]
# 
# cons <- paste(blurb_parts, collapse = " ") |>
#   str_replace_all("•", "") |>
#   str_squish()
# 
# 
# # Overview
# blurb_botton <- str_match(
#   block,
#   "\\bBottom line:\\s*([\\s\\S]*?)\\s*©"
# )[,2]
# 
# lines <- str_split(blurb_botton, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
# 
# blurb_parts <- unlist(segments)
# 
# position <- tail(blurb_parts, 1) |>
#   str_extract("[A-Z]{2,}$") |>
#   str_to_title()
# 
# blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
#   str_squish()
# blurb_parts <- blurb_parts[str_count(blurb_parts, "\\S+") > 3]
# 
# blurb_bottom_final <- paste(blurb_parts, collapse = " ") |>
#   str_squish()
# 
# block <- str_remove(block, "^\\s*PLAYER PROFILES\\s*")
# 
# 
# blurb_top <- str_match(
#   block,
#   "^\\s*([\\s\\S]*?)\\s*POSITION FIT"
# )[,2]
# 
# lines <- str_split(blurb_top, "\n")[[1]]
# segments <- str_split(lines, "\\s{10,}")
# blurb_parts <- unlist(segments)
# 
# full_name <- blurb_parts[
#   str_detect(blurb_parts, "^[A-Z\\s\\.\\-’']+$")
# ][1] |>
#   str_squish() |>
#   str_to_title()
# 
# 
# 
# blurb_parts_af <- blurb_parts[
#   str_detect(blurb_parts, "[a-z]") & 
#     !str_detect(blurb_parts, "^\\s*\\d")
# ] |>
#   str_squish()
# 
# blurb_top_final <- paste(blurb_parts_af, collapse = " ") |>
#   str_squish()
# 
# 
# blurb_final <- paste(blurb_bottom_final, blurb_top_final)

parse_prospect_2017 <- function(JackPlummer) {
  
  block <- paste(JackPlummer, collapse = "\n") |>
    str_replace_all("\\n+", "\n")
  
  # --- Pros ---
  blurb1 <- str_match(
    block,
    "(?i)\\bWhat he does best:\\s*([\\s\\S]*?)\\s*(?:Biggest concerns?|Areas of concern):"
  )[,2]
  
  lines <- str_split(blurb1, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
    str_squish()
  
  blurb_parts <- blurb_parts[
    str_detect(blurb_parts, "^\\s*•") | !str_detect(blurb_parts, "^\\s*[A-Z]")
  ]
  
  pros <- paste(blurb_parts, collapse = " ") |>
    str_replace_all("•", "") |>
    str_squish()
  
  # --- Cons ---
  cons1 <- str_match(
    block,
    "(?i)\\b(?:Biggest concerns?|Areas of concern):\\s*([\\s\\S]*?)\\s*(?:©|Bottom line:|Player comparison:)"
  )[,2]
  
  lines <- str_split(cons1, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
    str_squish()
  
  blurb_parts <- blurb_parts[
    str_detect(blurb_parts, "^\\s*•") | !str_detect(blurb_parts, "^\\s*[A-Z]")
  ]
  
  cons <- paste(blurb_parts, collapse = " ") |>
    str_replace_all("•", "") |>
    str_squish()
  
  # --- Bottom / overview ---
  blurb_botton <- str_match(
    block,
    "(?i)\\bBottom\\s+line:\\s*([\\s\\S]*?)\\s*(?:©|Combine\\s+results)"
  )[,2]
  
  lines <- str_split(blurb_botton, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  
  blurb_parts <- unlist(segments)
  
  position <- tail(blurb_parts, 1) |>
    str_extract("[A-Z]{2,}$") |>
    str_to_title()
  
  blurb_parts <- blurb_parts[str_detect(blurb_parts, "[a-z]")] |>
    str_replace_all("\\b[A-Z]{2,}\\b", "") |>
    str_squish()
  
  blurb_parts <- blurb_parts |>
    str_replace_all("^\\s*(?:\\d+\\s+)+", "")
  
  blurb_parts <- blurb_parts[str_count(blurb_parts, "\\S+") > 3]
  
  blurb_bottom_final <- paste(blurb_parts, collapse = " ") |>
    str_squish()
  
  # --- Top / name / intro ---
  block2 <- str_remove(block, "^\\s*PLAYER PROFILES\\s*")
  
  blurb_top <- str_match(
    block2,
    "^\\s*([\\s\\S]*?)\\s*POSITION FIT"
  )[,2]
  
  lines <- str_split(blurb_top, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  blurb_parts <- unlist(segments)
  
  full_name <- blurb_parts[
    str_detect(blurb_parts, "^[A-Z\\s\\.\\-’']+$")
  ][1] |>
    str_squish() |>
    str_to_title()
  message("Processing: ", full_name)
  
  blurb_parts_af <- blurb_parts[
    str_detect(blurb_parts, "[a-z]") &
      !str_detect(blurb_parts, "^\\s*\\d")
  ] |>
    str_squish()
  
  blurb_top_final <- paste(blurb_parts_af, collapse = " ") |>
    str_squish()
  
  if (str_detect(blurb_top_final, regex("bottom line:", ignore_case = TRUE))) {
    blurb_top_final <- ""
  }
  
  blurb_final <- paste0(blurb_bottom_final, blurb_top_final)
  
  tibble(
    full_name = full_name,
    position = position,
    overview = blurb_final,
    pros = pros,
    cons = cons
  )
}


start_pages <- pages[
  grepl("What he does best:", pages$text) &
    grepl("Biggest concern", pages$text),
]

start_idxs <- start_pages$page[start_pages$page <= 310]
n <- length(start_idxs)

prospect_blocks <- lapply(seq_along(start_idxs), function(k) {
  
  start <- start_idxs[k]
  
  end <- if (k < n) {
    start_idxs[k + 1] - 1
  } else {
    nrow(pages)
  }
  
  paste(pages$text[start:end], collapse = " ")
})


all_prospects <- lapply(prospect_blocks, parse_prospect_2017)
table_2017 <- do.call(rbind, all_prospects)

#write.csv(table_2017, "2017_data.csv")

# Zay jones comes up twice


# worst case seach in the bkurb for names that match and then match upon that



txt <- pdf_text("~/Documents/STSCI 4981/NLP Project with Varun/PFF 2025 NFL Draft Guide.pdf")


pages <- tibble(
  page = seq_along(txt),
  text = txt
)


JackPlummer <- pages[62:64,]
cat(JackPlummer$text)


block <- paste(JackPlummer, collapse = "\n") |>
  str_replace_all("\\n+", "\n")

cat(block)

blurb1 <- str_match(
  block,
  "^([^•]*)"
)[,2] |> str_squish()

blurb1_clean <- str_replace_all(blurb1, "\\\\n", " ") |> str_squish() |>
  str_replace("^.*?c\\(\\s*", "") |>
  str_replace("\\)$", "") |>
  str_trim(side = "left") |>  
  str_replace_all('"', "") |>
  str_replace_all("\\\\\"", "") |> 
  str_trim() 

blurb2 <- sub("^.*?•", "•", block)
lines <- str_split(blurb2, "\n")[[1]]
segments <- str_split(lines, "\\s{10,}")
blurb_parts <- unlist(segments)

block2 <- blurb_parts[grepl("^\\s*•", blurb_parts)]

block2_clean <- block2 |>
  str_split("\\\\n") |>
  unlist() |>
  str_trim()

blurb_parts_clean <- block2_clean |>
  str_replace_all("\\\\n", "") |>                 
  str_split("(?=•)") |>                         
  unlist() |>
  str_trim()

pros_and_cons <- blurb_parts_clean[grepl("^\\s*•", blurb_parts_clean)] |>
  (\(x) sub("^\\s*•\\s*", "", x))() |>
  (\(x) paste0(x, ".", collapse = " "))()


parse_prospect_2025 <- function(prospect_block) {
  
  block <- paste(prospect_block, collapse = "\n") |>
    str_replace_all("\\n+", "\n")
  
  # -----------------------------
  # Opening summary = from start to first bullet
  # -----------------------------
  blurb1 <- str_match(
    block,
    "^([^•]*)"
  )[,2]
  
  blurb1_clean <- blurb1 |>
    str_replace_all("\\\\n", " ") |>
    str_squish() |>
    str_replace("^.*?c\\(\\s*", "") |>
    str_replace("\\)$", "") |>
    str_trim(side = "left") |>
    str_replace_all('"', "") |>
    str_replace_all("\\\\\"", "") |>
    str_trim()
  
  # -----------------------------
  # Bullet section = from first bullet onward
  # -----------------------------
  blurb2 <- sub("^.*?•", "•", block)
  
  lines <- str_split(blurb2, "\n")[[1]]
  segments <- str_split(lines, "\\s{10,}")
  blurb_parts <- unlist(segments)
  
  block2 <- blurb_parts[grepl("^\\s*•", blurb_parts)]
  
  block2_clean <- block2 |>
    str_split("\\\\n") |>
    unlist() |>
    str_trim()
  
  blurb_parts_clean <- block2_clean |>
    str_replace_all("\\\\n", "") |>
    str_split("(?=•)") |>
    unlist() |>
    str_trim()
  
  pros_and_cons <- blurb_parts_clean[grepl("^\\s*•", blurb_parts_clean)] |>
    (\(x) sub("^\\s*•\\s*", "", x))() |>
    (\(x) sub("\\.*$", "", x))() |>
    (\(x) paste0(x, ".", collapse = " "))()
  
  tibble(
    summary = blurb1_clean,
    pros_and_cons = pros_and_cons
  )
}

start_pages <- pages |>
  filter(grepl("•", text))

start_idxs <- start_pages$page
n <- length(start_idxs)

prospect_blocks <- lapply(seq_along(start_idxs), function(k) {
  
  start <- start_idxs[k]
  
  end <- if (k < n) {
    start_idxs[k + 1] - 1
  } else {
    nrow(pages)
  }
  
  paste(pages$text[start:end], collapse = " ")
})


all_prospects <- lapply(prospect_blocks, parse_prospect_2025)
table_2025 <- do.call(rbind, all_prospects) |> 
  mutate(
    summary = str_replace(
      summary,
      "Welcome to PFF.*?150\\s*\\+\\s*player capsules with grades and",
      ""
    ) |>
      str_squish()
  ) |> 
  mutate(
    summary = str_squish(
      str_replace(.data$summary, "^advanced\\s*", "")
    )
  )
  
#write.csv(table_2025, "2025_data.csv")




# ---- helpers ----
library(rvest)
library(xml2)
library(dplyr)
library(purrr)
library(stringr)
library(tidyr)
library(readr)
library(tibble)

guide_url <- "https://www.pff.com/news/draft-2026-pff-nfl-draft-guide"

clean_text <- function(x) {
  x |>
    paste(collapse = " ") |>
    str_replace_all("\\u00a0", " ") |>
    str_replace_all("[[:space:]]+", " ") |>
    str_trim()
}

safe_read_html <- purrr::possibly(
  function(url) read_html(url),
  otherwise = NULL
)

clean_page <- function(page) {
  bad_nodes <- page |> html_elements("script, style, noscript, svg")
  if (length(bad_nodes) > 0) xml2::xml_remove(bad_nodes)
  page
}

get_player_links <- function(guide_url) {
  page <- safe_read_html(guide_url)
  if (is.null(page)) stop("Could not read guide page.")
  
  links <- page |>
    html_elements("a[href]") |>
    html_attr("href") |>
    unique()
  
  links <- links[!is.na(links)]
  links <- links[str_detect(links, "/news/draft-pff-2026-nfl-draft-guide-")]
  
  links <- ifelse(
    str_detect(links, "^https?://"),
    links,
    paste0("https://www.pff.com", links)
  )
  
  tibble(url = unique(links))
}

find_heading_node <- function(page, heading_text) {
  xpath <- paste0(
    "//*[self::h1 or self::h2 or self::h3 or self::h4]",
    "[translate(normalize-space(.), ",
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = '",
    tolower(heading_text), "']"
  )
  html_element(page, xpath = xpath)
}

extract_section_nodes <- function(page, heading_text) {
  heading <- find_heading_node(page, heading_text)
  if (length(heading) == 0 || is.na(heading)) return(list())
  
  out <- list()
  node <- xml2::xml_find_first(heading, "following-sibling::*[1]")
  
  while (length(node) > 0 && !inherits(node, "xml_missing")) {
    nm <- xml2::xml_name(node)
    
    if (nm %in% c("h1", "h2", "h3", "h4")) break
    
    out[[length(out) + 1]] <- node
    node <- xml2::xml_find_first(node, "following-sibling::*[1]")
  }
  
  out
}

extract_section_text <- function(page, heading_text) {
  nodes <- extract_section_nodes(page, heading_text)
  if (length(nodes) == 0) return(NA_character_)
  
  txt <- map_chr(nodes, ~ clean_text(html_text2(.x)))
  txt <- txt[txt != ""]
  
  txt <- txt[
    !str_detect(txt, "^@import\\b|^\\.pff-|^#pff-|^function\\(|^\\(function\\(") &
      !str_detect(txt, "Mock Draft Simulator|Big Board Builder|Explore PFF Tools|More Coverage")
  ]
  
  out <- clean_text(txt)
  
  out <- str_replace(out, regex("Player Traits[\\s\\S]*", ignore_case = TRUE), "")
  
  if (out == "") NA_character_ else out
}

extract_section_bullets <- function(page, heading_text) {
  nodes <- extract_section_nodes(page, heading_text)
  if (length(nodes) == 0) return(NA_character_)
  
  bullets <- c()
  
  for (node in nodes) {
    li_nodes <- html_elements(node, "li")
    
    if (length(li_nodes) > 0) {
      vals <- li_nodes |>
        html_text2() |>
        str_squish()
      bullets <- c(bullets, vals)
    } else {
      nm <- xml2::xml_name(node)
      if (nm %in% c("p", "div")) {
        val <- clean_text(html_text2(node))
        if (
          val != "" &&
          !str_detect(val, "^@import\\b|^\\.pff-|^#pff-|^function\\(|^\\(function\\(") &&
          !str_detect(val, "Player Traits|Season Stats|Mock Draft Simulator|Big Board Builder|More Coverage")
        ) {
          bullets <- c(bullets, val)
        }
      }
    }
  }
  
  bullets <- bullets[bullets != ""]
  bullets <- unique(bullets)
  
  out <- paste(bullets, collapse = " | ")
  if (out == "") NA_character_ else out
}

extract_player_comp <- function(page) {
  all_headings <- page |>
    html_elements("h1, h2, h3, h4, strong, b") |>
    html_text2() |>
    str_squish()
  
  comp_line <- all_headings[str_detect(all_headings, regex("^Player Comp\\s*:", ignore_case = TRUE))]
  
  if (length(comp_line) == 0) {
    body_text <- page |> html_text2() |> clean_text()
    comp_line <- str_match(body_text, regex("Player Comp\\s*:\\s*([^\\n]+)", ignore_case = TRUE))[,2]
    return(ifelse(is.na(comp_line), NA_character_, str_squish(comp_line)))
  }
  
  str_replace(comp_line[1], regex("^Player Comp\\s*:\\s*", ignore_case = TRUE), "") |>
    str_squish()
}

parse_player_page <- function(url) {
  page <- safe_read_html(url)
  
  if (is.null(page)) {
    return(tibble(
      player = NA_character_,
      school = NA_character_,
      position = NA_character_,
      scouting_report = NA_character_,
      strengths = NA_character_,
      weaknesses = NA_character_,
      bottom_line = NA_character_,
      comp = NA_character_
    ))
  }
  
  page <- clean_page(page)
  
  h1s <- page |> html_elements("h1") |> html_text2() |> str_squish()
  h1s <- h1s[h1s != ""]
  player <- if (length(h1s) >= 2) h1s[2] else if (length(h1s) >= 1) h1s[1] else NA_character_
  
  player <- player |>
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") |>
    str_squish()
  
  body_text <- page |> html_text2() |> clean_text()
  
  school <- str_match(body_text, paste0(stringr::fixed(player), "\\s+([A-Za-z .&'\\-]+)\\s+Height:"))[,2]
  position <- str_match(body_text, "Position:\\s*([A-Z/]+)")[,2]
  
  scouting_report <- extract_section_text(page, "Scouting report")
  scouting_report <- scouting_report |>
    str_split(regex("\\bPlayer Traits\\b", ignore_case = TRUE), n = 2) |>
    unlist() |> 
    str_squish()
  strengths       <- extract_section_bullets(page, "Strengths")
  weaknesses      <- extract_section_bullets(page, "Weaknesses")
  bottom_line     <- extract_section_text(page, "Bottom line")
  comp            <- extract_player_comp(page)
  
  tibble(
    player = player,
    school = school,
    position = position,
    scouting_report = scouting_report,
    strengths = strengths,
    weaknesses = weaknesses,
    bottom_line = bottom_line,
    comp = comp
  )
}
# ---- run ----

player_links <- get_player_links(guide_url)


fernando <- parse_player_page(
  "https://www.pff.com/news/draft-pff-2026-nfl-draft-guide-fernando-mendoza-indiana"
)

url <-  "https://www.pff.com/news/draft-pff-2026-nfl-draft-guide-fernando-mendoza-indiana"

cat("\nPLAYER:\n", fernando$player, "\n")
cat("\nSR:\n", fernando$scouting_report, "\n")
cat("\nSTRENGTHS:\n", fernando$strengths, "\n")
cat("\nWEAKNESSES:\n", fernando$weaknesses, "\n")
cat("\nBOTTOM LINE:\n", fernando$bottom_line, "\n")
cat("\nCOMP:\n", fernando$comp, "\n")

pff_2026_reports <- player_links |>
  mutate(data = map(url, parse_player_page)) |>
  unnest(data)

#write_csv(pff_2026_reports, "pff_2026_draft_reports.csv")

#pff_2026_reports


