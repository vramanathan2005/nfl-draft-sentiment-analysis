library(ggplot2)
library(dplyr)
library(scales)

# Load in ranked second 
ranked_second_contracts <- read.csv("ranked_second_contracts.csv") |> 
  select(-X)

# counts_rank <- 
ranked_second_contracts |> 
  ggplot() +
  geom_histogram(
    aes(x = rank_on_date_APY, fill = rank_on_date_APY == 1),
    binwidth = 1
  ) +
  scale_fill_manual(
    values = c("FALSE" = "darkgreen", "TRUE" = "#DC143C"),
    guide = "none"
  ) +
#  geom_histogram(aes(x = rank_on_date_APY), binwidth = 1, fill = "darkgreen", color = "darkgreen") +
    scale_x_continuous(
      breaks = seq(0, max(ranked_second_contracts$rank_on_date_APY, na.rm = TRUE), by = 10)
    ) +
  labs(
    title = "Distribution of APY Rank at Signing",
    x = "APY Rank on Signing Date",
    y = "Count"
  ) +
  theme_minimal() +
  theme(panel.grid.major.y = element_line(size = 0.5),
        plot.title = element_text(hjust=0.5, size=20),
        axis.text.y = element_text(colour="black")
          #plot.subtitle = element_text(hjust=0.5)
        )
  
# Why theoretically is there such a high number for first?
  # Well the answer is that if there is a possibility that a player is top 5 anyway, agents will make the push to get the player to have the highest paid contract


# We want to test now the distribution of players at a finer detail level
ranked_second_contracts |> 
  ggplot(aes(x = rank_on_date_APY)) +
  geom_histogram(
    breaks = c(1, 3, 8, 15, 25, 40, 60, 80, 150, Inf),
    fill = "darkgreen",
    color = "white"
  ) +
  scale_x_continuous(
    breaks = c(1, 3, 8, 15, 25, 40, 60, 80, 150)
  ) +
  scale_y_continuous(
    breaks = seq(0, 800, by = 100)
  ) +
  labs(
    title = "Distribution of APY Rank at Signing",
    subtitle = "APY Rank of 1-150 Included Only",
    x = "APY Rank on Signing Date",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major.y = element_line(linewidth = 0.5),
    plot.title = element_text(hjust = 0.5, size = 20),
    axis.text.y = element_text(colour = "black"),
    plot.subtitle = element_text(hjust=0.5)
  )

# This way we have good designations! 
# We have about 100 in each of the first 3 and then it increases a lot
# This way, 3, 5, 7, 10, 15, 20, and then 70 (or everything plus)



plot_group <- function(data, group_name) {
  data |>
    filter(side == group_name) |>
    ggplot(aes(x = rank_on_date_APY, y = APY, color = Position)) +
    geom_vline(
      # xintercept = c(1, 3, 8, 15, 25, 40, 60, 80),
      xintercept = c(1, 3, 8, 15, 25, 40, 80),
      linetype = "dashed",
      color = "grey60",
      linewidth = 0.4
    ) +
    # We want to use a GAM to smooth things out here
    # Cubic spline with shrinkage
    geom_smooth(
      method = "gam",
      formula = y ~ s(x, bs = "cs"),
      se = FALSE,
      linewidth = 1
    ) +
    scale_y_continuous(
      labels = scales::dollar_format(),
      limits = c(0, NA)
    ) +
    labs(
      title = paste("APY vs. APY Rank at Signing —", group_name),
      subtitle = "Only Top 150",
      x = "APY Rank on Signing Date",
      y = "APY",
      color = "Position"
    ) +
    theme_minimal()
}

ranked_second_contracts2 <- ranked_second_contracts |>
  mutate(
    side = case_when(
      Position %in% c("QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT") ~ "Offense",
      Position %in% c("EDGE", "DT", "LB", "CB", "S") ~ "Defense",
      Position %in% c("K", "P", "LS") ~ "Special Teams",
      TRUE ~ "Other"
    )
  ) |> 
  filter(rank_on_date_APY <= 150)

plot_group(ranked_second_contracts2, "Offense")

plot_group(ranked_second_contracts2, "Defense")

plot_group(ranked_second_contracts2, "Special Teams")


# Need labels!

# Maybe we are going with rank associated with percentile or percentage of max


# This was done to have ~53 players per roster
ranked_second_contracts_binned <- 
  ranked_second_contracts |> 
  mutate(
    rank_bin = cut(
      rank_on_date_APY,
      breaks = c(1, 3, 8, 15, 25, 40, 80, 110, Inf),
      include.lowest = TRUE,
      right = TRUE,
      labels = c("1-3", "4-8", "9-15", "16-25", "26-40", "41-80", "81-110", "111+")
    ),
    rank_label = cut(
      rank_on_date_APY,
      breaks = c(1, 3, 8, 15, 25, 40, 80, 110, Inf),
      include.lowest = TRUE,
      right = TRUE,
      labels = c("Superstar", "Star", "Above-Average Starter", "Starter", "Backup/Role-Player", "Depth", "Roster Fringe", "Off Roster")
    )
  )


# Next produce the same graph, but with the new labels!!!!
ranked_second_contracts_binned |> 
  ggplot(aes(x = rank_label)) +
  geom_bar(,
    fill = "darkgreen",
    color = "white"
  ) +
  scale_y_continuous(
    breaks = seq(0, 800, by = 100)
  ) +
  geom_text(
    stat = "count",
    aes(label = after_stat(count)),
    vjust = -0.3
  ) +
  labs(
    title = "Distribution of APY Rank at Signing",
    subtitle = "APY Rank of 1-150 Included Only",
    x = "APY Rank on Signing Date",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major.y = element_line(linewidth = 0.5),
    plot.title = element_text(hjust = 0.5, size = 20),
    axis.text.y = element_text(colour = "black"),
    plot.subtitle = element_text(hjust=0.5)
  )

# 52.1875
ranked_second_contracts_binned |>
  filter(rank_label %in% c("Superstar", "Star", "Above-Average Starter", "Starter", "Backup/Role-Player", "Depth", "Roster Fringe")) |>
  nrow()/32


# Can we get specific examples of who is in each?

# Need to find APY as a percentage of cap and find the ranges for each position (highest to lowest at each value or average?)

cap_ref <- read.csv("cap_ref.csv") |> select(cap, year)

ranked_second_contracts_apy <- 
  ranked_second_contracts_binned |> 
  mutate(signed_year = year(date_signed)) |> 
  left_join(free_agency_start_dates, by = c("signed_year" = "Year")) |> 
  mutate(
    season = if_else(date_signed < Date, signed_year - 1, signed_year))|>
  select(-signed_year, -Date) |> 
  left_join(cap_ref, by = c("season" = "year")) |> 
  mutate(apy_cap_pct = APY/(10^6)/cap)

# Now that have apy cap percentages we should find min and max for each position!



position_label_rank <-
  ranked_second_contracts_apy |> 
  group_by(rank_label, Position) |> 
  select(rank_label, Position, apy_cap_pct) |> 
  mutate(high_range = round(max(apy_cap_pct)*100, 3),
         low_range = round(min(apy_cap_pct)*100, 3),
    range = paste0("[", low_range, ", ", high_range, "]")) |> 
  select(-apy_cap_pct) |> ungroup() |> unique()
  



# Also give it salary ranges PER POSITION, WE WANT TO SEE!!!!



# By rank level we are looking
rank_levels <- c(
  "Superstar",
  "Star",
  "Above-Average Starter",
  "Starter",
  "Backup/Role-Player",
  "Depth",
  "Roster Fringe",
  "Off Roster"
)

pos_order <- position_label_rank |>
  filter(rank_label == "Superstar") |>
  arrange(desc(high_range)) |>
  pull(Position)

position_label_rank |>
  mutate(
    midpoint = (low_range + high_range) / 2,
    rank_label = factor(
      rank_label,
      levels = rank_levels
    ),
    Position = factor(Position, levels = rev(pos_order))
  ) |>
  filter(!is.na(Position)) |> 
  complete(
    Position,
    rank_label = factor(rank_levels, levels = rank_levels),
    fill = list(low_range = 0, high_range = 0)
  ) |> 
  ggplot(aes(y = Position, midpoint)) +
  geom_segment(
    aes(
      x = low_range,
      xend = high_range,
      yend = Position,
      color = rank_label
    ),
    linewidth = 3
  ) +
  geom_point(
    aes(x = midpoint, color = rank_label),
    size = 2
  ) +
  facet_wrap(~ rank_label, scales = "free_y") +
  labs(
    title = "APY as Percent of Cap by Position and Tier",
    x = "APY / Salary Cap (%)",
    y = "Position",
    color = "Tier"
  ) +
  theme_minimal()


# Offense only

offense_positions <- c("QB", "RB", "WR", "TE", "LT", "LG", "C", "RG", "RT")

pos_order2 <- position_label_rank |>
  filter(rank_label == "Superstar" & Position %in% offense_positions) |>
  arrange(desc(high_range)) |>
  pull(Position)

position_label_rank |>
  filter(Position %in% offense_positions) |>
  mutate(
    midpoint = (low_range + high_range) / 2,
    rank_label = factor(rank_label, levels = rank_levels),
    Position = factor(Position, levels = pos_order2)
  ) |>
  complete(
    Position,
    rank_label = factor(rank_levels, levels = rev(rank_levels)),
    fill = list(low_range = 0, high_range = 0)
  ) |>
  mutate(
    midpoint = (low_range + high_range) / 2
  ) |>
  ggplot(aes(y = rank_label)) +
  geom_segment(
    aes(
      x = low_range,
      xend = high_range,
      yend = rank_label,
      color = Position
    ),
    linewidth = 3
  ) +
  geom_point(
    aes(x = midpoint, color = Position),
    size = 2
  ) +
  facet_wrap(~ Position, scales = "free_x") +
  labs(
    title = "Offensive APY as Percent of Cap by Position and Tier",
    x = "APY / Salary Cap (%)",
    y = "Tier",
    color = "Position"
  ) +
  theme_minimal()



### OL grouped together
offense_groups <- c("QB", "RB", "WR", "TE", "OL")

position_label_rank_offense <- position_label_rank |>
  mutate(
    offense_group = case_when(
      Position %in% c("LT", "LG", "C", "RG", "RT") ~ "OL",
      Position %in% c("QB", "RB", "WR", "TE") ~ Position,
      TRUE ~ NA_character_
    )
  ) |>
  filter(!is.na(offense_group)) |>
  group_by(offense_group, rank_label) |>
  summarise(
    low_range = min(low_range, na.rm = TRUE),
    high_range = max(high_range, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    midpoint = (low_range + high_range) / 2,
    rank_label = factor(rank_label, levels = rank_levels),
    offense_group = factor(offense_group, levels = offense_groups)
  )

position_label_rank_offense |>
  complete(
    offense_group,
    rank_label = factor(rank_levels, levels = rev(rank_levels)),
    fill = list(low_range = 0, high_range = 0)
  ) |>
  mutate(
    midpoint = (low_range + high_range) / 2
  ) |>
  ggplot(aes(y = rank_label)) +
  geom_segment(
    aes(
      x = low_range,
      xend = high_range,
      yend = rank_label,
      color = offense_group
    ),
    linewidth = 3
  ) +
  geom_point(
    aes(x = midpoint, color = offense_group),
    size = 2
  ) +
  facet_wrap(~ offense_group, scales = "free_x") +
  labs(
    title = "Offensive APY as Percent of Cap by Position Group and Tier",
    x = "APY / Salary Cap (%)",
    y = "Tier",
    color = "Position Group"
  ) +
  theme_minimal()

# Making OL one big thing does not seem to smooth things out, check offensive vs defensive after 


# Defensive Graphs
defense_positions <- c("EDGE", "DT", "LB", "CB", "S")

pos_order3 <- position_label_rank |>
  filter(rank_label == "Superstar" & Position %in% defense_positions) |>
  arrange(desc(high_range)) |>
  pull(Position)

position_label_rank |>
  filter(Position %in% defense_positions) |>
  mutate(
    midpoint = (low_range + high_range) / 2,
    rank_label = factor(rank_label, levels = rank_levels),
    Position = factor(Position, levels = pos_order3)
  ) |>
  complete(
    Position,
    rank_label = factor(rank_levels, levels = rev(rank_levels)),
    fill = list(low_range = 0, high_range = 0)
  ) |>
  mutate(
    midpoint = (low_range + high_range) / 2
  ) |>
  ggplot(aes(y = rank_label)) +
  geom_segment(
    aes(
      x = low_range,
      xend = high_range,
      yend = rank_label,
      color = Position
    ),
    linewidth = 3
  ) +
  geom_point(
    aes(x = midpoint, color = Position),
    size = 2
  ) +
  facet_wrap(~ Position, scales = "free_x") +
  labs(
    title = "Offensive APY as Percent of Cap by Position and Tier",
    x = "APY / Salary Cap (%)",
    y = "Tier",
    color = "Position"
  ) +
  theme_minimal()

# Defensive curves seem to be less pronounced than offensive ones, meaning second contract non-superstars are making more




# Offense vs defense total
side_positions <- c(
  "QB", "RB", "WR", "TE", "LT", "LG", "C", "RG", "RT",
  "EDGE", "DT", "LB", "CB", "S"
)


side_ranges <- position_label_rank |>
  filter(Position %in% side_positions) |>
  mutate(
    side = case_when(
      Position %in% c("QB", "RB", "WR", "TE", "LT", "LG", "C", "RG", "RT") ~ "Offense",
      Position %in% c("EDGE", "DT", "LB", "CB", "S") ~ "Defense"
    )
  ) |>
  group_by(side, rank_label) |>
  summarise(
    low_range = min(low_range, na.rm = TRUE),
    high_range = max(high_range, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    midpoint = (low_range + high_range) / 2,
    rank_label = factor(rank_label, levels = rev(rank_levels))
  )

side_ranges |>
  ggplot(aes(y = rank_label)) +
  geom_segment(
    aes(
      x = low_range,
      xend = high_range,
      yend = rank_label,
      color = side
    ),
    linewidth = 8,
    alpha = 0.45
  ) +
  geom_point(
    aes(x = midpoint, color = side),
    size = 3,
    alpha = 0.8
  ) +
  scale_color_manual(
    values = c(
      "Offense" = "darkred",
      "Defense" = "darkblue"
    )
  ) +
  labs(
    title = "Offense vs. Defense APY Cap % Ranges by Tier",
    x = "APY / Salary Cap (%)",
    y = "Tier",
    color = "Side"
  ) +
  theme_minimal()


# Offensive superstars and stars (top 8) get the high end of salary cap in terms of what they make, 
# while defensive players in starter or depth roles tend to frequently make more as a percentage of cap
# If you're not one of the highest paid players, you might want to be a defensive player





# Maybe you can find a way to rank draftpicks based on their second contracts (need percentage of max contract at the time)
# use pick value to judge things (like where they could rank-- later picked players get more value than earlier).
# but also include apy because that distingueshes positional value!
# can there be a negative weight for historically good picks at a position?

