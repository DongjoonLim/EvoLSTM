from Bio import AlignIO
import bx.align.maf
'''
for alignment in bx.align.maf.Reader(open("chrY.anc.maf")):
    print (alignment.split())
'''
for multiple_alignment in AlignIO.parse("chrY.anc.maf", "maf"):
    print("printing a new multiple alignment")

    for seqrec in multiple_alignment:
        print("starts at %s on the %s strand of a sequence %s in length, and runs for %s bp" % \
              (seqrec.annotations["start"],
               seqrec.annotations["strand"],
               seqrec.annotations["srcSize"],
               seqrec.annotations["size"]))