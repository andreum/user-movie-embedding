import pandas as pd

#
# german-scores 
#
# $ grep ,10915, clean-ratings-3.csv > x2
# $ grep ,9698, clean-ratings-3.csv >> x2
# $ cut -d, -f1 x2 | sort | uniq -c | cut -c1-8 | sort | uniq -c
#   8126       1 
#   2126       2 
# $ cut -d, -f2 x2 | uniq -c
#   4943 9698
#   7435 10915
#
# Just Downfall: 4943 - 2126 = 2817
# Just Lives of Others: 7435 - 2126 = 5309
# Both Downfall and LoO: 2126

# 
# cat x2 | perl -ne 'chomp $_; @a = split/,/; $x{$a[0]}{$a[1]} = $a[2]; END {for $user (keys %x) {@films = keys %{$x{$user}}; @ratings = map { ($x{$user}{$_})} @films; printf "%s\n", join(",", @ratings) if @ratings == 2}}' > german-scores
#

sw = pd.read_csv("german-scores", header = None)
print(sw.corr())
#         0        1
#0  1.00000  0.32893
#1  0.32893  1.00000
