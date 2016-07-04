# The MIT License (MIT)
# 
#   Copyright (c) 2016 Wei-Hung Weng
# 
#   Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#   
#   The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE. 
# 
# Title : Active Learning function
# Author : Wei-Hung Weng
# Created : 07/01/2016
# Comment : 

# get top guess item

training <- readRDS("~/Desktop/lmrExp/lmrNeuroBin/analysis/training_1.rds")
inTraining <- createDataPartition(training$yLabel, p=0.7, list=FALSE)
tr <- training[inTraining, ]
te <- training[-inTraining, ]
X <- tr
X$yLabel <- NULL
y <- tr[, c("yLabel")]
y <- replace(y, -c(1:100), NA)

# http://www.slideshare.net/web2webs/active-learning-scenarios-and-techniques
# https://www.cs.utah.edu/~piyush/teaching/10-11-print.pdf

UncertaintySampling <- function(X, y, uncertainty="entropy", classifier="svmLinear", num_query=3) { # classifier must be in caret
  X <- as.matrix(X)
  y <- factor(y)
  idx_unlabeled <- which(is.na(y))
  split <- list(X[-idx_unlabeled, ], y[-idx_unlabeled], X[idx_unlabeled, ], y[idx_unlabeled])
  
  control <- trainControl(method="none", savePredictions=TRUE,
                          allowParallel=FALSE, classProbs=TRUE)
  model <- train(x=split[[1]], y=split[[2]], method=classifier, trControl=control, verbose=TRUE, metric="accuracy")
  posterior <- predict(model, split[[3]], type="prob") # pred table with only probability
  posterior <- as.matrix(posterior)
  
  # computes the uncertainty for each of the unlabeled observations
  if (uncertainty == "entropy") {
    pred <- apply(posterior, 1, entropy.plugin) # entropy package
  } else if (uncertainty == "least") {
    pred <- apply(posterior, 1, max)
  } else if (uncertainty == "margin") {
    pred <- apply(posterior, 1, function(post_i) {post_i[order(post_i, decreasing=T)[1:2]] %*% c(1, -1)})
  } else {
    stop("Should select")
  }
  
  # Determines the order of the unlabeled observations by uncertainty measure
  query <- order(pred, decreasing=TRUE)[seq_len(num_query)]
  list(query=query, posterior=posterior, uncertainty=obs_uncertainty)
}

UncertaintySampling(X, y, "least", "svmLinear", 5)$query






QueryRandom <- function(y, num=1) {
  idx_unlabeled <- which(is.na(y))
  if (length(idx_unlabeled) <= 1 | length(idx_unlabeled) <= num) {
    query <- idx_unlabeled
  } else {
    query <- sample(idx_unlabeled, num)
  }
  list(query=query, unlabeled=idx_unlabeled)
}

QueryRandom(y, 5)$query





QueryCommittee <- function(X, y, classifier=c("svmLinear", "svmLinear"), disagreement="post", num_query=1) {
  library(entropy)
  X <- as.matrix(X)
  y <- factor(y)
  idx_unlabeled <- which(is.na(y))
  split <- list(X[-idx_unlabeled, ], y[-idx_unlabeled], X[idx_unlabeled, ], y[idx_unlabeled])
  
  # Classifies the unlabeled observations with each committee member and also
  # determines their posterior probabilities.
  committee <- list()
  control <- trainControl(method="none", savePredictions=TRUE,
                          allowParallel=FALSE, classProbs=TRUE)
  for (i in 1:length(classifier)) {
    model <- train(x=split[[1]], y=split[[2]], method=classifier[i], trControl=control, verbose=TRUE, metric="accuracy")
    posterior <- predict(model, split[[3]], type="prob") # pred table with only probability
    committee[[i]] <- as.matrix(posterior)
  }
  
  if (disagreement == "vote") { # vote entropy
    for (i in 1:length(committee)) {
      for (j in 1:nrow(posterior)) {
        committee[[i]][j, 1] <- ifelse(committee[[i]][j, 1] > committee[[i]][j, 2], 1, 0)
        committee[[i]][j, 2] <- ifelse(committee[[i]][j, 1] > committee[[i]][j, 2], 0, 1)
      }
    }
    vote <- Reduce('+', committee) / length(committee)
    d <- apply(avg, 1, function(x) entropy.plugin(x))
  } else if (disagreement == "post") { # post entropy
    avg <- Reduce('+', committee) / length(committee)
    d <- apply(avg, 1, function(x) entropy.plugin(x))
  } else if (disagreement == "kullback") { # KL divergence
    avg <- Reduce('+', committee) / length(committee)
    k <- lapply(committee, function(x) rowSums(x * log(x / avg)))
    d <- Reduce('+', k) / length(k)
  } else {
    stop("Should select")
  }
    
  query <- order(d, decreasing=TRUE)[seq_len(num_query)]
  list(query=query, disagreement=disagreement, committee_predictions=committee_predictions)
}


QueryCommittee(X, y, classifier, "post", 5)$query


