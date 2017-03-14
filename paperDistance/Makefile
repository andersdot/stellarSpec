LATEX       = pdflatex
CHECK_RERUN = grep "Rerun to get" $*.log

all: paper.pdf

%.pdf: %.tex lmanders.bib
	${LATEX} $*
	bibtex $*
	${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
