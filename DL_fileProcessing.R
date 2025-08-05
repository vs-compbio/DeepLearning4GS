### DL_Paper data processing 
install.packages("data.table")  # Install the package (only needed once)
library(data.table)

genofull<-fread("/uufs/chpc.utah.edu/common/home/akaundal-group1/vishal/Projects/GS_most/GS_corn_phd/gs_data_processing/fullGenoDataWithRowndColunNameProcessedSNPinRowsJul13.csv", header=TRUE)
genofull<-as.data.frame(genofull)
genofull[1:10, 1:10]
nrow(genofull)

rownames(genofull)<-genofull[,1]
rownames(genofull)
genofull<-genofull[,-1]
phenofull399<-read.csv("FulldatasetPheno.csv", row.names = 1)

# Keeping 399 in genofile
library(dplyr)

commonnames<- intersect(colnames(genofull), rownames(phenofull399))
geno399<-genofull %>%
  select(all_of(commonnames))


ncol(geno399)
geno399[1:10, 1:10]
str(geno399)
sum(is.na(geno399))
sum(geno399=="")
geno399nomiss<-subset(geno399, !apply(geno399, 1, function(x) any(x=="")))
sum(geno399nomiss=="")
nrow(geno399nomiss)
geno399nomiss[1:10, 1:10]
library(WGCNA)
geno396_t<-transposeBigData(geno399nomiss, blocksize = 20000)
geno396_t[1:10, 1:10]

#fwrite(geno399nomiss, "geno_396_794k.csv")


# snpReady
library(snpReady)
geno.ready2 <- raw.data(data = as.matrix(geno396_t), frame = "wide", base = TRUE, sweep.sample = 0.5, call.rate = 0.95, maf = 0.05, imput = FALSE, outfile = "012")
M<-geno.ready2$M.clean
M[1:10, 1:10]
rownames(M)
M<-as.data.frame(M)

#fwrite(M, "geno396_349k012.csv", row.names = TRUE)

# 100K 
set.seed(123)
geno396_100kcol<-sample(ncol(M), 100000)
geno396_100k<-M[,geno396_100kcol]
geno396_100k[1:10, 1:10]
# 50k
set.seed(1234)
geno396_50kcol<-sample(ncol(geno396_100k), 50000)
geno396_50k<-geno396_100k[,geno396_50kcol]
geno396_50k[1:10, 1:10]

# 25K
set.seed(12345)
geno396_25kcol<-sample(ncol(geno396_50k), 25000)
geno396_25k<-geno396_50k[,geno396_25kcol]
geno396_25k[1:10, 1:10]

# 10K
set.seed(123456)
geno396_10kcol<-sample(ncol(geno396_25k), 10000)
geno396_10k<-geno396_25k[,geno396_10kcol]
geno396_10k[1:10, 1:10]
ncol(geno396_10k)

# 5K
set.seed(1234567)
geno396_5kcol<-sample(ncol(geno396_10k), 5000)
geno396_5k<-geno396_10k[,geno396_5kcol]
geno396_5k[1:10, 1:10]
ncol(geno396_5k)

## merging pheno with geno
pheno_STIall<-read.csv("pheno_STIall.csv", row.names = 1)


nrow(pheno_STIall)
commonRnames<- intersect(rownames(pheno_STIall), rownames(geno396_5k))
pheno_STIall <- pheno_STIall[commonRnames, , drop = FALSE]

# merge pheno and geno data
geno396_100k_ph<-merge(geno396_100k, pheno_STIall, by="row.names", all=TRUE)
geno396_100k_ph[1:10, 1:15]
rownames(geno396_100k_ph) <- geno396_100k_ph$Row.names
geno396_100k_ph <- geno396_100k_ph[, -1]

geno396_50k_ph<-merge(geno396_50k, pheno_STIall, by="row.names", all=TRUE)
geno396_50k_ph[1:10, 1:10]
rownames(geno396_50k_ph) <- geno396_50k_ph$Row.names
geno396_50k_ph <- geno396_50k_ph[, -1]

geno396_25k_ph<-merge(geno396_25k, pheno_STIall, by="row.names", all=TRUE)
geno396_25k_ph[1:10, 1:10]
rownames(geno396_25k_ph) <- geno396_25k_ph$Row.names
geno396_25k_ph <- geno396_25k_ph[, -1]

geno396_10k_ph<-merge(geno396_10k, pheno_STIall, by="row.names", all=TRUE)
rownames(geno396_10k_ph) <- geno396_10k_ph$Row.names
geno396_10k_ph <- geno396_10k_ph[, -1]
geno396_10k_ph[1:10, 1:10]

geno396_5k_ph<-merge(geno396_5k, pheno_STIall, by="row.names", all=TRUE)
rownames(geno396_5k_ph) <- geno396_5k_ph$Row.names
geno396_5k_ph <- geno396_5k_ph[, -1]
geno396_5k_ph[1:5, 1:5]

write.csv(geno396_100k_ph, "geno396_100k_ph.csv")
write.csv(geno396_50k_ph, "geno396_50k_ph.csv")
write.csv(geno396_25k_ph, "geno396_25k_ph.csv")
write.csv(geno396_10k_ph, "geno396_10k_ph.csv")
write.csv(geno396_5k_ph, "geno396_5k_ph.csv")



