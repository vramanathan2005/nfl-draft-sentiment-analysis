library(dplyr)
library(lubridate)
library(purrr)
library(dplyr)
library(stringr)
library(nflreadr)

nfl_contracts <- load_contracts() |> #filter(is_active == T) |> 
  select(player, otc_id, draft_year) |> unique()

# Do the above with the APY and such just for purposes of the contracts 
# signed between december 2024 and now, but be careful about exact dates

OTC_contacts <- read.csv("OTC_Contracts.csv")

OTC_contracts_clean <- OTC_contacts |> 
  mutate(
    date_signed = mdy(date_signed),
    date_ended = na_if(date_ended, "0000-00-00"),
    date_ended = mdy(date_ended)
  ) |> 
  select(ID, player_id, team_id, Position, contract_type, date_signed, date_ended, 
         APY, Years, Total, Guarantee, total_guarantees, signing_bonus) |> 
  left_join(nfl_contracts, by = c("player_id" = "otc_id"), relationship = "many-to-many") |> 
  distinct() |> 
  mutate(
    date_signed = ymd(date_signed),
    date_ended = ymd(date_ended)
  ) |> 
  arrange(player_id, date_signed) |> 
  group_by(player_id) |> 
  mutate(
    date_ended = if_else(
      !is.na(lead(date_signed)),
      lead(date_signed),
      date_ended
    )
  ) |> 
  ungroup()

# before 2017 contracts signed don't matter
free_agency_start_dates <- tibble(
  Year = c(2014:2024),
  Date = as.Date(c(
    "2014-03-11",
    "2015-03-10",
    "2016-03-09",
    "2017-03-09",
    "2018-03-14",
    "2019-03-13",
    "2020-03-18",
    "2021-03-17",
    "2022-03-16",
    "2023-03-15",
    "2024-03-13",
  ))
)

cutoff_date <- as.Date("2024-11-14")


OTC_contracts_clean2 <- OTC_contracts_clean |>
  mutate(
    date_signed = as.Date(date_signed),
    date_ended  = as.Date(date_ended),
    exp_year = year(date_signed) + Years
  ) |>
  left_join(
    free_agency_start_dates,
    by = c("exp_year" = "Year")
  ) |>
  mutate(
    date_ended = case_when(
      is.na(date_signed) ~ date_ended,
      is.na(date_ended) & !is.na(Date) & Date < cutoff_date ~ Date,
      TRUE ~ date_ended
    )
  ) |>
  select(-Date) |> 
  filter(exp_year > 2017)









#Players like calvin johnson who retire are issue


contracts_ranked <- OTC_contracts_clean2  |> 
  mutate(
    date_signed = ymd(date_signed),
    date_ended = ymd(date_ended)
  ) |>
  filter(!is.na(date_signed)) |>
  filter(is.na(date_ended) | date_ended > date_signed) |>
  distinct(
    player_id, contract_type, #team_id, 
    contract_type, date_signed, APY, Total, Guarantee,
    .keep_all = TRUE
  ) |>
  arrange(date_signed) |>
  mutate(row_id = row_number())


# rank_table <- map_dfr(seq_len(nrow(contracts_ranked)), function(i) {
#   
#   signing_day <- contracts_ranked$date_signed[i]
#   this_apy <- contracts_ranked$APY[i]
#   this_total <- contracts_ranked$Total[i]
#   this_guarantee <- contracts_ranked$Guarantee[i]
#   
#   active_contracts <- contracts_ranked |>
#     filter(
#       date_signed <= signing_day,
#       is.na(date_ended) | date_ended >= signing_day
#     )
#   
#   tibble(
#     row_id = contracts_ranked$row_id[i],
#     rank_on_date_APY = min_rank(desc(c(active_contracts$APY, this_apy)))[length(active_contracts$APY) + 1],
#     rank_on_date_Total = min_rank(desc(c(active_contracts$Total, this_total)))[length(active_contracts$Total) + 1],
#     rank_on_date_Guarantee = min_rank(desc(c(active_contracts$Guarantee, this_guarantee)))[length(active_contracts$Guarantee) + 1],
#     active_n = nrow(active_contracts)
#   )
# })

contracts_ranked <- contracts_ranked %>%
  mutate(row_id = row_number())

contracts_by_pos <- split(contracts_ranked, contracts_ranked$Position)

rank_table <- map_dfr(names(contracts_by_pos), function(pos) {
  
  df <- contracts_by_pos[[pos]]
  
  map_dfr(unique(df$date_signed), function(signing_day) {
    
    active_contracts <- df %>%
      filter(
        date_signed <= signing_day,
        is.na(date_ended) | date_ended >= signing_day
      )
    
    focal <- df %>%
      filter(date_signed == signing_day)
    
    focal %>%
      mutate(
        rank_on_date_APY = sapply(APY, function(x) {
          vals <- c(active_contracts$APY, x)
          min_rank(desc(vals))[length(vals)]
        }),
        rank_on_date_Total = sapply(Total, function(x) {
          vals <- c(active_contracts$Total, x)
          min_rank(desc(vals))[length(vals)]
        }),
        rank_on_date_Guarantee = sapply(Guarantee, function(x) {
          vals <- c(active_contracts$Guarantee, x)
          min_rank(desc(vals))[length(vals)]
        }),
        active_n = nrow(active_contracts)
      ) %>%
      select(row_id, rank_on_date_APY, rank_on_date_Total, rank_on_date_Guarantee, active_n)
    
  })
})


contracts_ranked <- contracts_ranked %>%
  left_join(rank_table, by = "row_id") %>%
  select(-row_id)



contracts_ranked_important <- contracts_ranked |> 
  select(player, team_id, Position, date_signed, date_ended, contract_type, APY, Years, draft_year, rank_on_date_APY) |> 
  unique() |> 
  filter(draft_year >= 2017) |> 
  filter(!(Position %in% c("T", "G", "NULL"))) |> 
  select(!c(team_id, date_ended)) |> unique()


contracts_ranked_important_flag <- contracts_ranked_important |> 
  group_by(player, Position) |> 
  filter(date_signed >= "2017-5-01") |> 
  mutate(contract_num = row_number()) |> 
  filter(contract_num == 2) |> 
  filter(Position != "IDL") # This was just an extra position for a random irrelevent player, not even enough work to make specifically not





files <- list.files(
  pattern = "^consensus-big-board-\\d{4}-\\d{4}\\.csv$",
  full.names = TRUE
)

big_boards_all <- map_dfr(files, function(f) {
  draft_year <- str_extract(basename(f), "\\d{4}") %>% as.integer()
  
  read_csv(f, show_col_types = FALSE) %>%
    mutate(draft_year = draft_year)
})

big_boards_all <- big_boards_all |> mutate(`Player Name` = clean_player_names(`Player Name`))


all_draft_picks <- load_draft_picks(2017:2026) |> select(pfr_player_name, gsis_id, round, pick, season) |> 
  mutate(pfr_player_name = clean_player_names(pfr_player_name))



big_boards_all_pick_real <- big_boards_all |> 
  mutate(
    `Player Name` = case_when(
      `Player Name` == "Jalen Tabor" ~ "Teez Tabor",
      `Player Name` == "Ogbo Okoronkwo" ~ "Ogbo Okoronkwo",
      `Player Name` == "Cameron Skattebo" ~ "Cam Skattebo",
      `Player Name` == "Cameron Ward" ~ "Cam Ward",
      `Player Name` == "Josh Allen" & draft_year == 2019 ~ "Josh Hines-Allen",
      `Player Name` == "Jeffrey Okudah" ~ "Jeff Okudah",
      `Player Name` == "Justin Madubuike" ~ "Nnamdi Madubuike",
      `Player Name` == "Trevon Moehrig-Woodard" ~ "Trevon Moehrig",
      `Player Name` == "Cameron Jurgens" ~ "Cam Jurgens",
      `Player Name` == "Henry Tootoo" ~ "Henry TooToo",
      `Player Name` == "Byron Young (TN)" ~ "Byron Young",
      `Player Name` == "Obo Okoronkwo" ~ "Ogbonnia Okoronkwo",
      `Player Name` == "Demeioun Robinson" ~ "Chop Robinson",
      TRUE ~ `Player Name`
    )
  ) |> 
  rename(consensus = Rank) |> 
  left_join(all_draft_picks |> 
              mutate(pfr_player_name = case_when(
                gsis_id == "00-0038978" ~ "Byron Young (AL)",
                TRUE ~ pfr_player_name
              )), by = c("Player Name" = "pfr_player_name", "draft_year" = "season"))

na_player_bad <- big_boards_all_pick_real |> 
  filter(consensus < 200 & draft_year < 2026 & is.na(pick))
  

# Use LLM to check if any of these players were drafted, if it is wrong can afford the error
#write.csv(big_boards_all_pick_real, "big_boards_all_pick_real.csv")

max_picks_by_year <- big_boards_all_pick_real |> 
  filter(!is.na(pick)) |> 
  group_by(draft_year) |> 
  summarise(max_pick = max(pick, na.rm = TRUE), .groups = "drop") |> 
  add_row(
    draft_year = 2026,
    max_pick = 257
  )

big_board_fall <- big_boards_all_pick_real |> 
  left_join(max_picks_by_year, by = "draft_year") |> 
  mutate(below_consensus = consensus - pick) |> 
  mutate(below_consensus = case_when(
    is.na(pick) & consensus <= max_pick & draft_year < 2026 ~ "undrafted - proj drafted",
    is.na(pick) & consensus > max_pick & draft_year < 2026 ~ "undrafted - proj undrafted",
    TRUE ~ as.character(below_consensus)
  )) |> 
  mutate(pick = ifelse(is.na(pick) & draft_year < 2026, "undrafted", pick))

### Clean the prospects in consensus

#big_board_fall <- read.csv("~/Documents/STSCI 4981/NLP Project with Varun/player_consensus.csv")




#write.csv(big_board_fall, "player_consensus.csv")
#write.csv(contracts_ranked_important_flag, "ranked_second_contracts.csv")
  
  

### Clean the prospects in consensus

# Only for other stuff
# contracts_ranked_important_interesting <- contracts_ranked_important %>%
#   group_by(player, date_signed, rank_on_date_APY) %>%
#   filter((n() > 1 & date_signed == date_ended)) %>%
#   ungroup()
  

### Now that we have this, we are GOING TO HAVE TO FIND 2nd extensions
# Goes up through 21


