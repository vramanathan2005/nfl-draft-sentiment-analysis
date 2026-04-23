
library(tidyverse)

all_prospect_notes <- read.csv("all_prospects.csv") 


all_prospect_notes_OT_2026 <- 
  all_prospect_notes |> 
  filter(Position == "OT" & draft_year == 2026)


print_scouting_report <- function(df, player_name) {
  
  row <- df[df$Player.Name == player_name, ]
  
  if (nrow(row) == 0) {
    cat("Player not found\n")
    return(NULL)
  }
  
  cat("\n==============================\n")
  cat("SCOUTING REPORT:", player_name, "\n")
  cat("==============================\n\n")
  
  # SCOUTING REPORT
  if (!is.na(row$beast_summary)) {
    cat("SCOUTING REPORT:\n")
    cat(row$beast_summary, "\n\n")
  }
  
  # STRENGTHS
  if (!is.na(row$beast_strengths)) {
    strengths <- unlist(strsplit(row$beast_strengths, "●|\\|"))
    strengths <- trimws(strengths)
    strengths <- strengths[strengths != ""]
    
    cat("STRENGTHS:\n")
    cat(paste0("- ", strengths), sep = "\n")
    cat("\n\n")
  }
  
  # WEAKNESSES
  if (!is.na(row$beast_weaknesses)) {
    weaknesses <- unlist(strsplit(row$beast_weaknesses, "●|\\|"))
    weaknesses <- trimws(weaknesses)
    weaknesses <- weaknesses[weaknesses != ""]
    
    cat("WEAKNESSES:\n")
    cat(paste0("- ", weaknesses), sep = "\n")
    cat("\n")
  }
}


print_scouting_report(all_prospect_notes_OT_2026, "Caleb Lomu")





inference_2026 <- read.csv("inference_2026.csv") 

