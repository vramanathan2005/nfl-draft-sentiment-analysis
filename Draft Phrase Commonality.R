library(tidyverse)
library(tidytext)

inference_2026 <- read.csv("inference_2026.csv") 

# clean_words <- function(text) {
#   tibble(text = text) |>
#     unnest_tokens(word, text) |>
#     filter(!word %in% stop_words$word,
#            str_detect(word, "[a-z]")) |>
#     distinct(word) |>
#     pull(word)
# }
# 
# player <- all_prospect_notes |>
#   filter(Player.Name == "Fernando Mendoza", draft_year == 2026)
# 
# beast_txt <- paste(
#   player$beast_summary,
#   player$beast_strengths,
#   player$beast_weaknesses,
#   collapse = " "
# )
# 
# pff_txt <- paste(
#   player$pff_overview,
#   player$pff_pros,
#   player$pff_cons,
#   player$pff_bottom_line,
#   player$pff_extra,
#   collapse = " "
# )
# 
# br_txt <- paste(
#   player$br_positives,
#   player$br_negatives,
#   collapse = " "
# )
# 
# 
# clean_bigrams_counts <- function(text) {
#   tibble(text = text) |>
#     unnest_tokens(bigram, text, token = "ngrams", n = 2) |>
#     separate(bigram, into = c("w1", "w2"), sep = " ") |>
#     filter(!w1 %in% stop_words$word,
#            !w2 %in% stop_words$word,
#            str_detect(w1, "[a-z]"),
#            str_detect(w2, "[a-z]")) |>
#     unite(bigram, w1, w2, sep = " ") |>
#     count(bigram, sort = TRUE)
# }
# 
# beast_counts <- clean_bigrams_counts(beast_txt) |> mutate(source = "beast")
# pff_counts   <- clean_bigrams_counts(pff_txt)   |> mutate(source = "pff")
# br_counts    <- clean_bigrams_counts(br_txt)    |> mutate(source = "br")
# 
# all_counts <- bind_rows(beast_counts, pff_counts, br_counts)
# 
# bigram_summary <- all_counts |>
#   group_by(bigram) |>
#   summarise(
#     total_count = sum(n),              # total occurrences across all text
#     sources_present = n_distinct(source),  # how many sources it appears in
#     beast_n = sum(n[source == "beast"], na.rm = TRUE),
#     pff_n   = sum(n[source == "pff"],   na.rm = TRUE),
#     br_n    = sum(n[source == "br"],    na.rm = TRUE),
#     .groups = "drop"
#   ) |>
#   arrange(desc(sources_present), desc(total_count))
# 
# common_2plus_counts <- bigram_summary |>
#   filter(sources_present >= 2, total_count > 1)
# 
# common_2plus_counts


clean_ngram_counts <- function(text, n = 2) {
  cols <- paste0("w", 1:n)
  
  tibble(text = text) |>
    unnest_tokens(ngram, text, token = "ngrams", n = n) |>
    separate(ngram, into = cols, sep = " ", remove = TRUE) |>
    filter(if_all(all_of(cols), ~ !.x %in% stop_words$word),
           if_all(all_of(cols), ~ str_detect(.x, "[a-z]"))) |>
    unite("ngram", all_of(cols), sep = " ") |>
    count(ngram, sort = TRUE)
}



get_common_ngrams_player <- function(player_row, n = 2) {
  
  beast_txt <- paste(
    player_row$beast_summary,
    player_row$beast_strengths,
    player_row$beast_weaknesses,
    collapse = " "
  )
  
  pff_txt <- paste(
    player_row$pff_overview,
    player_row$pff_pros,
    player_row$pff_cons,
    player_row$pff_bottom_line,
    player_row$pff_extra,
    collapse = " "
  )
  
  br_txt <- paste(
    player_row$br_positives,
    player_row$br_negatives,
    collapse = " "
  )
  
  beast_counts <- clean_ngram_counts(beast_txt, n = n) |> mutate(source = "beast")
  pff_counts   <- clean_ngram_counts(pff_txt,   n = n) |> mutate(source = "pff")
  br_counts    <- clean_ngram_counts(br_txt,    n = n) |> mutate(source = "br")
  
  all_counts <- bind_rows(beast_counts, pff_counts, br_counts)
  
  if (nrow(all_counts) == 0) {
    return(tibble())
  }
  
  all_counts |>
    group_by(ngram) |>
    summarise(
      total_count = sum(n),
      sources_present = n_distinct(source),
      beast_n = sum(n[source == "beast"], na.rm = TRUE),
      pff_n   = sum(n[source == "pff"],   na.rm = TRUE),
      br_n    = sum(n[source == "br"],    na.rm = TRUE),
      .groups = "drop"
    ) |>
    filter(sources_present >= 2, total_count > 1) |>
    mutate(
      Player.Name = player_row$Player.Name[1],
      Position    = player_row$Position[1],
      ngram_size  = n
    ) |>
    select(Player.Name, Position, ngram_size, everything())
}


remove_contained_ngrams <- function(df) {
  keep <- rep(TRUE, nrow(df))
  
  for (i in seq_len(nrow(df))) {
    shorter <- df$ngram[i]
    shorter_n <- df$ngram_size[i]
    
    for (j in seq_len(nrow(df))) {
      if (i == j) next
      
      longer <- df$ngram[j]
      longer_n <- df$ngram_size[j]
      
      if (shorter_n < longer_n) {
        pattern <- paste0("\\b", stringr::str_replace_all(shorter, " ", "\\\\s+"), "\\b")
        if (stringr::str_detect(longer, pattern)) {
          keep[i] <- FALSE
          break
        }
      }
    }
  }
  
  df[keep, ]
}

get_phrase_table_player <- function(player_row, n_values = 2:7) {
  
  phrase_df <- purrr::map_dfr(
    n_values,
    ~ get_common_ngrams_player(player_row, n = .x)
  )
  
  if (nrow(phrase_df) == 0) {
    return(tibble())
  }
  
  phrase_df |>
    arrange(desc(ngram_size), desc(sources_present), desc(total_count)) |>
    remove_contained_ngrams() |>
    arrange(desc(ngram_size), desc(sources_present), desc(total_count))
}

fernando <- inference_2026 |>
  filter(Player.Name == "Fernando Mendoza")

fernando_table <- get_phrase_table_player(fernando)

fernando_table


all_players_phrase_table_ngrams <- inference_2026 |>
  group_by(Player.Name, Position) |>
  group_split() |>
  purrr::map_dfr(get_phrase_table_player)


all_players_phrase_table_ngrams


# The highest number is 9!!! For leveon moss, we have to make sure to move some things around with the website!
all_players_phrase_table_ngrams |>
  count(Player.Name, sort = TRUE)

#write.csv(all_players_phrase_table_ngrams, "2026_draft_ngrams.csv")


