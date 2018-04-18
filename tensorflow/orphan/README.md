orphan.tab columns:
 * ps
 * length
 * exon.count
 * pI
 * masked
 * GC
 * transmembrane.domains
 * coils
 * rem465
 * hotloops


```R
require(readr)
d <- read_tsv("all-tips-data.tab")
d <- d[, c("ps", "length", "exon.count", "pI",
           "masked", "GC", "transmembrane.domains",
           "ncomp", "coils", "rem465", "hotloops")]
d <- d[!is.na(d$transmembrane.domains), ]
d <- d[!is.na(d$ncomp), ]
# ensure none of these are missing
summary(d)
# scale everything to be between 0 and 1
ps = d$ps
d <- as.data.frame(apply(d, 2, function(x) (x - mean(x)) / sd(x)))
d$ps = ps
write.table(d[2001:nrow(d), ], file="train.csv", quote=FALSE, sep=",", row.names=FALSE)
write.table(d[1:2000, ], file="test.csv", quote=FALSE, sep=",", row.names=FALSE)
```
