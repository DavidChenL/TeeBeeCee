
# Kelvin Fung
# Explore articles.json

library(rjson)
library(tidyverse)  # For dplyr and stringr
library(tm)  # For vector of stopwords (`stopwords()`)
library(e1071)  # For Naive Bayes Model
library(rpart)   # For decision tree.
library(rpart.plot)  # For decision tree.
json_data <- fromJSON(file = "articles.json")

num_art <- length(json_data)  # No. of articles.

### Generating a binary outcome based on
### 1) FB engagement count,
### 2) Max velocity,
### 3) Twitter shares.

# Helper function to normalise.
normalise <- function(vector) {
  vec_sd <- sd(vector)
  (vector - mean(vector)) / vec_sd 
}

# Extract FB engagement count.
all_fb_engmnt_ct <- numeric(num_art)  # Initialise vector for fb count.
for (i in seq_along(all_fb_engmnt_ct)) {
  all_fb_engmnt_ct[i] <- json_data[[i]]$fb_data$total_engagement_count
}
norm_all_fb_engmnt_ct <- (all_fb_engmnt_ct)^(1/3)  # Transforming by cube root.
norm_all_fb_engmnt_ct <- normalise(all_fb_engmnt_ct) # Normalise.
summary(norm_all_fb_engmnt_ct)

# Extract max velocity.
all_max_velo <- numeric(num_art)
for (i in seq_along(all_max_velo)) {
  all_max_velo[i] <- json_data[[i]]$max_velocity
}
norm_all_max_velo <- sqrt(all_max_velo)  # Transforming by square root.
norm_all_max_velo <- normalise(all_max_velo)  # Normalise.
summary(norm_all_max_velo)

# Extract twitter shares.
all_tw_shares <- numeric(num_art)
for (i in seq_along(all_tw_shares)) {
  all_tw_shares[i] <- json_data[[i]]$tw_data$tw_count
}
norm_all_tw_shares <- sqrt(all_tw_shares)
norm_all_tw_shares <- normalise(all_tw_shares)
summary(norm_all_tw_shares)

### ------------------------------------------------

### Extracting more predictor variables' data to build dataframe.

# Extract has_video.
all_has_vid <- logical(num_art)
for (i in seq_along(all_has_vid)) {
  all_has_vid[i] <- json_data[[i]]$has_video
}

# Extract publisher.
all_publisher <- character(num_art)
for (i in seq_along(all_publisher)) {
  all_publisher[i] <- json_data[[i]]$source$publisher
}

# Extract sentiment
all_sentiment <- numeric(num_art)
for (i in seq_along(all_sentiment)) {
  all_sentiment[i] <- json_data[[i]]$sentiment
}

# Build data frame with relevant predictor variables.
articles <- data.frame(all_fb_engmnt_ct,
                       all_max_velo,
                       all_tw_shares,
                       all_has_vid,
                       all_publisher,
                       all_sentiment)

# Bind our metric, which is the sum of normalised fb engagement, 
# max velocity and twitter share.
articles$fb_maxvelo_tw <- 
  norm_all_fb_engmnt_ct + 
  norm_all_max_velo +
  norm_all_tw_shares

# Find 85th quantile of metric.
thres_85 <- quantile(articles$fb_maxvelo_tw, 0.85)

articles$outcome <- articles$fb_maxvelo_tw >= thres_85
write_csv(articles, path = "articles_df_for_pred_md.csv")

### END OF DATA PREPARATION ----------------------------

### Train Naive Bayes Model 
df <- read_csv("articles_df_for_pred_md.csv")
df$all_publisher <- NULL
df$fb_maxvelo_tw <- NULL

model <- naiveBayes(outcome ~ .,
                    data = df)

model

test_df <- df[sample(39109, 1), ]

test_df <- test_df %>%
  rename(fb_engagement = all_fb_engmnt_ct,
         twitter_shares = all_tw_shares,
         max_velocity = all_max_velo,
         video = all_has_vid)

test_df$fb_engagement <- 93
test_df$twitter_shares <- 20
test_df$max_velocity <- 117
test_df$video <- TRUE
test_df$outcome <- NULL
test_df$all_sentiment <- -1

test_df[, 1:5] <- c(93, 117, 20, TRUE, -1)
test_df$all_has_vid <- TRUE

results <- unname(predict(model, test_df))

dt_fit <- rpart(outcome ~ .,
                data = df,
                method = "class",
                control=rpart.control(minsplit= 1),
                parms=list(split='information'))

rpart.plot(dt_fit, type = 4, extra = 1)
