pyftsubset 1570788235.ttf --output-file=1570788235-subset.ttf --text=" "`printf $(printf '\\%03o' $(seq 33 126))``cat ../src/*.rs | perl -CIO -pe 's/[\p{ASCII} \N{U+2500}-\N{U+257F}]//g'`
