r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
install.packages("sentometrics")
install.packages("lexicon")
install.packages("repmis")
install.packages("stm")
IRkernel::installspec(name='maibi_ba', prefix='${VSC_HOME}/.local')
