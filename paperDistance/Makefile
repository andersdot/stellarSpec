LATEX       = pdflatex
CHECK_RERUN = grep "Rerun to get" $*.log

all: paper.pdf

%.pdf: %.tex
	${LATEX} $*
	bibtex $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
