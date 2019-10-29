#Required packages
library(rjson)
library(tm)
library(SnowballC)
library(wordcloud)
library(dplyr)
library(tidytext)
library(data.table)

setwd("#INSERT PATH TO DIRECTORY")

#Reading json and using a sample
result <- fromJSON(file = "articles.json")

df <- data.frame(content = character(), sentiment = integer(), stringsAsFactors = FALSE)

for (i in 1:length(result)){
  if (result[[i]]$fb_data$total_engagement_count >= 100 & result[[i]]$velocity >= 0.05){
    df[i,1] <- result[[i]]$contents
    df[i,2] <- result[[i]]$sentiment
  }
}

df <- na.omit(df)

documents <- Corpus(VectorSource(df$content))

documents <- tm_map(documents, content_transformer(function (x , regex) gsub(regex, " ", x)), "[^a-zA-Z0-9]")

documents <- documents %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, c(stopwords("english"), "the")) %>%
  tm_map(stripWhitespace)

#Finding top 10 common words
x <- sort(rowSums(as.matrix(TermDocumentMatrix(documents))), decreasing = TRUE)
wordfreq <- data.frame(word = names(x),freq = x)
head(wordfreq, 10)
commonwords <- as.vector(wordfreq$word[1:10])

documents <- documents %>%
  tm_map(removeWords, commonwords)

x_lesscommon <- sort(rowSums(as.matrix(TermDocumentMatrix(documents))), decreasing = TRUE)
wordfreq_lesscommon <- data.frame(words = names(x_lesscommon),frequency = x_lesscommon)

#Generic wordcloud

wordcloud(words = wordfreq_lesscommon$words, freq = wordfreq_lesscommon$frequency, min.freq = 1,
          max.words=50, random.order=FALSE, rot.per=0.20, 
          colors=brewer.pal(5, "Blues"))

#Neutral wordcloud

x_lesscommon <- sort(rowSums(as.matrix(TermDocumentMatrix(documents[df$sentiment == 0]))), decreasing = TRUE)
wordfreq_lesscommon <- data.frame(words = names(x_lesscommon),frequency = x_lesscommon)

wordcloud(words = wordfreq_lesscommon$words, freq = wordfreq_lesscommon$frequency, min.freq = 1,
          max.words=50, random.order=FALSE, rot.per=0.20, 
          colors=brewer.pal(5, "Blues"))

#Positive wordcloud

x_lesscommon <- sort(rowSums(as.matrix(TermDocumentMatrix(documents[df$sentiment == 1]))), decreasing = TRUE)
wordfreq_lesscommon <- data.frame(words = names(x_lesscommon),frequency = x_lesscommon)

wordcloud(words = wordfreq_lesscommon$words, freq = wordfreq_lesscommon$frequency, min.freq = 1,
          max.words=50, random.order=FALSE, rot.per=0.20, 
          colors=brewer.pal(5, "Blues"))

#Negative wordcloud

x_lesscommon <- sort(rowSums(as.matrix(TermDocumentMatrix(documents[df$sentiment == -1]))), decreasing = TRUE)
wordfreq_lesscommon <- data.frame(words = names(x_lesscommon),frequency = x_lesscommon)

wordcloud(words = wordfreq_lesscommon$words, freq = wordfreq_lesscommon$frequency, min.freq = 1,
          max.words=50, random.order=FALSE, rot.per=0.20, 
          colors=brewer.pal(5, "Blues"))
