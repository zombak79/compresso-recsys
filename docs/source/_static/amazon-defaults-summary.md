# Verified Amazon default profiles

Audit completed: 2026-09-09T12:44:50.053752+00:00. All 33 categories / 132 split profiles are installed and verified.

Support means minimum user/item interactions. Users/items are preprocessed counts for user, item and leave-last-out splits; for temporal they are the unique user union and final cumulative catalog. Training pairs are unique training user-item interactions. Val/test cells show eligible users / distinct target items / cold target-item percentage (relative to the original training matrix). † flags fewer than 1,000 eligible users in either evaluation phase.

All ratings are binary interactions; preprocessing uses iterative support filtering only, without subsampling. Seed 42, eval_draws=1. Clothing temporal alone has the approved 120,000-user / 25,000-item cap exception. Books/Electronics use their larger caps and 20,000 requested users per user-split evaluation partition.

| Category | Split | Support U/I | Users | Items | Train pairs | Val: users / items / cold | Test: users / items / cold | Warning |
|---|---|---:|---:|---:|---:|---|---|---|
| All_Beauty | user_split | 2/2 | 22,634 | 12,105 | 45,509 | 2,097 / 1,826 / 0.00% | 2,068 / 1,854 / 0.00% |  |
| All_Beauty | item_split | 2/2 | 22,634 | 12,105 | 48,222 | 2,414 / 602 / 100.00% | 4,813 / 1,203 / 100.00% |  |
| All_Beauty | leave_last_out | 4/1 | 2,909 | 10,266 | 13,878 | 2,909 / 2,503 / 60.53% | 2,909 / 2,487 / 64.41% |  |
| All_Beauty | temporal | 3/1 | 3,622 | 10,734 | 12,649 | 1,410 / 1,657 / 81.11% | 670 / 618 / 89.97% | † Small evaluation set |
| Amazon_Fashion | user_split | 2/4 | 35,714 | 10,217 | 68,051 | 3,551 / 2,779 / 0.00% | 3,560 / 2,692 / 0.00% |  |
| Amazon_Fashion | item_split | 2/4 | 35,714 | 10,217 | 72,875 | 3,762 / 511 / 100.00% | 6,772 / 1,022 / 100.00% |  |
| Amazon_Fashion | leave_last_out | 4/2 | 1,717 | 4,098 | 11,044 | 1,717 / 1,292 / 15.48% | 1,717 / 1,190 / 21.18% |  |
| Amazon_Fashion | temporal | 2/2 | 3,157 | 4,080 | 8,888 | 1,116 / 673 / 72.07% | 791 / 467 / 80.09% | † Small evaluation set |
| Appliances | user_split | 3/2 | 52,101 | 19,327 | 153,605 | 4,872 / 3,391 / 0.00% | 4,859 / 3,277 / 0.00% |  |
| Appliances | item_split | 3/2 | 52,101 | 19,327 | 160,268 | 9,980 / 967 / 100.00% | 16,170 / 1,933 / 100.00% |  |
| Appliances | leave_last_out | 4/2 | 14,826 | 10,896 | 43,016 | 14,826 / 6,274 / 14.87% | 14,826 / 6,215 / 17.28% |  |
| Appliances | temporal | 2/2 | 71,639 | 16,820 | 71,406 | 28,766 / 8,307 / 37.52% | 20,484 / 7,199 / 47.84% |  |
| Arts_Crafts_and_Sewing | user_split | 5/16 | 97,533 | 18,043 | 728,532 | 5,000 / 6,506 / 0.00% | 5,000 / 6,613 / 0.00% |  |
| Arts_Crafts_and_Sewing | item_split | 5/16 | 97,533 | 18,043 | 689,186 | 31,814 / 903 / 100.00% | 52,915 / 1,805 / 100.00% |  |
| Arts_Crafts_and_Sewing | leave_last_out | 5/16 | 97,533 | 18,043 | 617,052 | 97,533 / 16,863 / 0.00% | 97,533 / 16,459 / 0.00% |  |
| Arts_Crafts_and_Sewing | temporal | 3/12 | 96,749 | 12,713 | 252,131 | 42,979 / 9,307 / 21.85% | 29,693 / 9,221 / 37.64% |  |
| Automotive | user_split | 7/25 | 99,264 | 16,079 | 1,009,231 | 5,000 / 7,492 / 0.00% | 5,000 / 7,574 / 0.00% |  |
| Automotive | item_split | 7/25 | 99,264 | 16,079 | 948,329 | 42,440 / 804 / 100.00% | 66,549 / 1,608 / 100.00% |  |
| Automotive | leave_last_out | 7/25 | 99,264 | 16,079 | 924,973 | 99,264 / 15,101 / 0.00% | 99,264 / 14,639 / 0.00% |  |
| Automotive | temporal | 6/12 | 86,556 | 19,182 | 428,612 | 49,082 / 15,181 / 25.95% | 30,194 / 12,891 / 29.71% |  |
| Baby_Products | user_split | 6/9 | 85,240 | 17,987 | 719,963 | 5,000 / 6,029 / 0.00% | 5,000 / 6,056 / 0.00% |  |
| Baby_Products | item_split | 6/9 | 85,240 | 17,987 | 697,358 | 29,585 / 900 / 100.00% | 49,964 / 1,799 / 100.00% |  |
| Baby_Products | leave_last_out | 6/9 | 85,240 | 17,987 | 645,022 | 85,240 / 14,115 / 0.03% | 85,240 / 13,560 / 0.03% |  |
| Baby_Products | temporal | 3/7 | 95,735 | 13,365 | 263,221 | 38,591 / 7,822 / 22.00% | 25,313 / 8,198 / 43.40% |  |
| Beauty_and_Personal_Care | user_split | 8/25 | 95,840 | 16,817 | 1,100,549 | 5,000 / 7,981 / 0.00% | 5,000 / 8,008 / 0.00% |  |
| Beauty_and_Personal_Care | item_split | 8/25 | 95,840 | 16,817 | 1,042,505 | 44,765 / 841 / 100.00% | 66,388 / 1,682 / 100.00% |  |
| Beauty_and_Personal_Care | leave_last_out | 8/25 | 95,840 | 16,817 | 1,039,531 | 95,840 / 15,180 / 0.00% | 95,840 / 14,750 / 0.00% |  |
| Beauty_and_Personal_Care | temporal | 6/18 | 95,044 | 14,656 | 392,445 | 50,440 / 9,685 / 28.90% | 40,215 / 11,789 / 48.75% |  |
| Books | user_split | 5/17 | 428,352 | 94,864 | 4,457,061 | 20,000 / 31,567 / 0.00% | 20,000 / 32,453 / 0.00% |  |
| Books | item_split | 5/17 | 428,352 | 94,864 | 4,194,262 | 161,770 / 4,744 / 100.00% | 241,076 / 9,487 / 100.00% |  |
| Books | leave_last_out | 5/17 | 428,352 | 94,864 | 4,054,971 | 428,352 / 80,327 / 0.04% | 428,352 / 76,778 / 0.05% |  |
| Books | temporal | 6/6 | 112,433 | 96,465 | 1,336,615 | 63,581 / 37,923 / 28.87% | 41,812 / 26,194 / 40.21% |  |
| CDs_and_Vinyl | user_split | 5/16 | 66,625 | 18,586 | 658,391 | 5,000 / 8,295 / 0.00% | 5,000 / 8,493 / 0.00% |  |
| CDs_and_Vinyl | item_split | 5/16 | 66,625 | 18,586 | 662,593 | 24,546 / 930 / 100.00% | 38,473 / 1,859 / 100.00% |  |
| CDs_and_Vinyl | leave_last_out | 5/16 | 66,625 | 18,586 | 643,324 | 66,625 / 16,271 / <0.01% | 66,625 / 16,039 / 0.01% |  |
| CDs_and_Vinyl | temporal | 2/5 | 29,237 | 13,650 | 109,653 | 13,120 / 6,986 / 22.70% | 6,481 / 4,566 / 24.11% |  |
| Cell_Phones_and_Accessories | user_split | 6/13 | 92,781 | 19,322 | 676,209 | 5,000 / 6,692 / 0.00% | 5,000 / 6,736 / 0.00% |  |
| Cell_Phones_and_Accessories | item_split | 6/13 | 92,781 | 19,322 | 646,767 | 28,616 / 967 / 100.00% | 52,067 / 1,933 / 100.00% |  |
| Cell_Phones_and_Accessories | leave_last_out | 6/13 | 92,781 | 19,322 | 572,962 | 92,781 / 16,504 / 0.04% | 92,781 / 15,069 / 0.05% |  |
| Cell_Phones_and_Accessories | temporal | 5/8 | 89,713 | 19,096 | 232,405 | 41,928 / 7,932 / 38.87% | 35,885 / 10,338 / 66.97% |  |
| Clothing_Shoes_and_Jewelry | user_split | 10/35 | 93,809 | 14,289 | 1,176,756 | 5,000 / 8,145 / 0.00% | 5,000 / 8,166 / 0.00% |  |
| Clothing_Shoes_and_Jewelry | item_split | 10/35 | 93,809 | 14,289 | 1,118,190 | 46,057 / 715 / 100.00% | 70,344 / 1,429 / 100.00% |  |
| Clothing_Shoes_and_Jewelry | leave_last_out | 10/35 | 93,809 | 14,289 | 1,129,762 | 93,809 / 13,220 / 0.00% | 93,809 / 12,877 / 0.00% |  |
| Clothing_Shoes_and_Jewelry | temporal | 10/17 | 113,384 | 24,824 | 428,731 | 76,251 / 20,663 / 54.84% | 59,629 / 19,421 / 57.30% |  |
| Digital_Music | user_split | 2/2 | 2,782 | 2,389 | 5,996 | 254 / 252 / 0.00% | 244 / 240 / 0.00% | † Small evaluation set |
| Digital_Music | item_split | 2/2 | 2,782 | 2,389 | 6,392 | 314 / 119 / 100.00% | 563 / 236 / 100.00% | † Small evaluation set |
| Digital_Music | leave_last_out | 4/1 | 2,243 | 16,240 | 13,823 | 2,243 / 2,184 / 85.21% | 2,243 / 2,177 / 86.08% |  |
| Digital_Music | temporal | 2/1 | 2,115 | 8,154 | 4,417 | 862 / 1,121 / 92.51% | 604 / 735 / 93.61% | † Small evaluation set |
| Electronics | user_split | 8/16 | 494,089 | 92,180 | 6,276,852 | 20,000 / 31,288 / 0.00% | 20,000 / 31,489 / 0.00% |  |
| Electronics | item_split | 8/16 | 494,089 | 92,180 | 5,833,634 | 224,865 / 4,609 / 100.00% | 347,160 / 9,218 / 100.00% |  |
| Electronics | leave_last_out | 8/16 | 494,089 | 92,180 | 5,844,291 | 494,089 / 72,119 / <0.01% | 494,089 / 66,509 / <0.01% |  |
| Electronics | temporal | 6/12 | 484,391 | 78,239 | 2,936,581 | 252,303 / 39,800 / 21.04% | 177,638 / 40,647 / 40.37% |  |
| Gift_Cards | user_split | 2/1 | 11,426 | 749 | 22,960 | 1,135 / 230 / 0.00% | 1,131 / 240 / 0.00% |  |
| Gift_Cards | item_split | 2/2 | 11,333 | 575 | 24,323 | 2,285 / 29 / 100.00% | 1,445 / 58 / 100.00% |  |
| Gift_Cards | leave_last_out | 4/1 | 1,201 | 503 | 3,853 | 1,201 / 253 / 14.62% | 1,201 / 244 / 14.75% |  |
| Gift_Cards | temporal | 2/1 | 2,313 | 421 | 2,266 | 1,092 / 123 / 28.46% | 577 / 120 / 38.33% | † Small evaluation set |
| Grocery_and_Gourmet_Food | user_split | 7/24 | 98,300 | 16,735 | 1,097,509 | 5,000 / 7,768 / 0.00% | 5,000 / 7,813 / 0.00% |  |
| Grocery_and_Gourmet_Food | item_split | 7/24 | 98,300 | 16,735 | 1,045,821 | 42,173 / 837 / 100.00% | 63,925 / 1,674 / 100.00% |  |
| Grocery_and_Gourmet_Food | leave_last_out | 7/24 | 98,300 | 16,735 | 1,026,671 | 98,300 / 15,347 / 0.00% | 98,300 / 14,851 / 0.00% |  |
| Grocery_and_Gourmet_Food | temporal | 6/13 | 86,148 | 19,238 | 438,683 | 50,360 / 14,478 / 32.93% | 34,731 / 13,429 / 39.73% |  |
| Handmade_Products | user_split | 2/2 | 22,156 | 11,878 | 40,876 | 2,006 / 1,725 / 0.00% | 2,040 / 1,788 / 0.00% |  |
| Handmade_Products | item_split | 2/2 | 22,156 | 11,878 | 43,229 | 2,335 / 587 / 100.00% | 4,283 / 1,180 / 100.00% |  |
| Handmade_Products | leave_last_out | 4/1 | 3,716 | 14,723 | 12,001 | 3,716 / 3,350 / 76.24% | 3,716 / 3,347 / 77.14% |  |
| Handmade_Products | temporal | 3/1 | 3,774 | 11,710 | 5,823 | 1,410 / 2,029 / 88.66% | 1,251 / 1,708 / 92.62% |  |
| Health_and_Household | user_split | 10/20 | 84,204 | 19,922 | 1,177,657 | 5,000 / 9,180 / 0.00% | 5,000 / 8,986 / 0.00% |  |
| Health_and_Household | item_split | 10/20 | 84,204 | 19,922 | 1,137,763 | 44,334 / 997 / 100.00% | 63,720 / 1,993 / 100.00% |  |
| Health_and_Household | leave_last_out | 10/20 | 84,204 | 19,922 | 1,166,630 | 84,204 / 16,439 / 0.00% | 84,204 / 15,703 / 0.00% |  |
| Health_and_Household | temporal | 8/15 | 88,598 | 19,907 | 552,121 | 50,052 / 13,166 / 26.50% | 38,620 / 14,545 / 44.23% |  |
| Health_and_Personal_Care | user_split | 2/1 | 18,396 | 17,288 | 35,887 | 1,041 / 940 / 0.00% | 1,038 / 877 / 0.00% |  |
| Health_and_Personal_Care | item_split | 2/1 | 18,396 | 17,288 | 38,004 | 1,758 / 784 / 100.00% | 3,890 / 1,585 / 100.00% |  |
| Health_and_Personal_Care | leave_last_out | 4/1 | 1,109 | 3,448 | 6,372 | 1,109 / 911 / 42.26% | 1,109 / 857 / 47.37% |  |
| Health_and_Personal_Care | temporal | 2/1 | 5,291 | 7,604 | 9,108 | 2,100 / 1,603 / 72.99% | 1,040 / 762 / 77.30% |  |
| Home_and_Kitchen | user_split | 12/35 | 96,909 | 17,431 | 1,536,242 | 5,000 / 9,721 / 0.00% | 5,000 / 9,750 / 0.00% |  |
| Home_and_Kitchen | item_split | 12/35 | 96,909 | 17,431 | 1,465,801 | 51,957 / 872 / 100.00% | 78,055 / 1,744 / 100.00% |  |
| Home_and_Kitchen | leave_last_out | 12/35 | 96,909 | 17,431 | 1,518,169 | 96,909 / 15,181 / 0.00% | 96,909 / 14,374 / 0.00% |  |
| Home_and_Kitchen | temporal | 10/25 | 93,903 | 14,879 | 712,464 | 53,997 / 11,264 / 20.23% | 38,614 / 10,726 / 28.82% |  |
| Industrial_and_Scientific | user_split | 4/8 | 74,487 | 19,251 | 407,624 | 5,000 / 5,348 / 0.00% | 5,000 / 5,350 / 0.00% |  |
| Industrial_and_Scientific | item_split | 4/8 | 74,487 | 19,251 | 403,192 | 17,942 / 963 / 100.00% | 33,333 / 1,926 / 100.00% |  |
| Industrial_and_Scientific | leave_last_out | 4/8 | 74,487 | 19,251 | 322,305 | 74,487 / 16,742 / 0.12% | 74,487 / 16,000 / 0.14% |  |
| Industrial_and_Scientific | temporal | 3/5 | 66,901 | 17,690 | 134,812 | 31,809 / 10,279 / 36.56% | 23,291 / 10,034 / 52.33% |  |
| Kindle_Store | user_split | 12/76 | 98,841 | 17,800 | 2,607,203 | 5,000 / 13,209 / 0.00% | 5,000 / 13,406 / 0.00% |  |
| Kindle_Store | item_split | 12/76 | 98,841 | 17,800 | 2,471,905 | 63,226 / 890 / 100.00% | 85,505 / 1,780 / 100.00% |  |
| Kindle_Store | leave_last_out | 12/76 | 98,841 | 17,800 | 2,705,614 | 98,841 / 15,961 / 0.00% | 98,841 / 15,154 / 0.00% |  |
| Kindle_Store | temporal | 8/50 | 96,853 | 19,957 | 1,680,921 | 60,020 / 15,954 / 16.75% | 42,975 / 14,155 / 24.56% |  |
| Magazine_Subscriptions | user_split | 2/1 | 6,808 | 2,003 | 11,414 | 1,115 / 472 / 0.00% | 1,093 / 463 / 0.00% |  |
| Magazine_Subscriptions | item_split | 2/2 | 6,408 | 1,241 | 13,334 | 1,070 / 63 / 100.00% | 1,594 / 124 / 100.00% |  |
| Magazine_Subscriptions | leave_last_out | 4/1 | 797 | 1,110 | 2,822 | 797 / 353 / 25.21% | 797 / 381 / 30.45% | † Small evaluation set |
| Magazine_Subscriptions | temporal | 2/1 | 458 | 522 | 830 | 151 / 117 / 45.30% | 74 / 66 / 33.33% | † Small evaluation set |
| Movies_and_TV | user_split | 10/36 | 97,871 | 18,324 | 1,619,663 | 5,000 / 10,254 / 0.00% | 5,000 / 10,070 / 0.00% |  |
| Movies_and_TV | item_split | 10/36 | 97,871 | 18,324 | 1,539,020 | 50,964 / 917 / 100.00% | 76,392 / 1,833 / 100.00% |  |
| Movies_and_TV | leave_last_out | 10/36 | 97,871 | 18,324 | 1,606,685 | 97,871 / 16,454 / 0.00% | 97,871 / 15,927 / 0.00% |  |
| Movies_and_TV | temporal | 5/12 | 60,122 | 17,522 | 422,446 | 31,406 / 10,612 / 18.37% | 16,203 / 6,941 / 23.51% |  |
| Musical_Instruments | user_split | 5/7 | 50,431 | 16,766 | 354,789 | 5,000 / 5,933 / 0.00% | 5,000 / 5,863 / 0.00% |  |
| Musical_Instruments | item_split | 5/7 | 50,431 | 16,766 | 375,422 | 18,353 / 839 / 100.00% | 27,033 / 1,677 / 100.00% |  |
| Musical_Instruments | leave_last_out | 5/7 | 50,431 | 16,766 | 341,225 | 50,431 / 12,985 / 0.05% | 50,431 / 12,681 / 0.05% |  |
| Musical_Instruments | temporal | 2/5 | 77,684 | 15,914 | 178,287 | 32,616 / 10,123 / 23.50% | 23,272 / 9,093 / 33.53% |  |
| Office_Products | user_split | 5/18 | 98,570 | 12,740 | 659,700 | 5,000 / 5,177 / 0.00% | 5,000 / 5,178 / 0.00% |  |
| Office_Products | item_split | 5/18 | 98,570 | 12,740 | 625,565 | 30,907 / 637 / 100.00% | 49,343 / 1,274 / 100.00% |  |
| Office_Products | leave_last_out | 5/18 | 98,570 | 12,740 | 537,042 | 98,570 / 12,194 / 0.02% | 98,570 / 11,905 / 0.02% |  |
| Office_Products | temporal | 4/8 | 90,062 | 19,965 | 275,927 | 39,715 / 11,142 / 22.83% | 29,909 / 12,535 / 47.53% |  |
| Patio_Lawn_and_Garden | user_split | 6/21 | 98,613 | 14,019 | 777,521 | 5,000 / 6,556 / 0.00% | 5,000 / 6,539 / 0.00% |  |
| Patio_Lawn_and_Garden | item_split | 6/21 | 98,613 | 14,019 | 739,989 | 33,334 / 701 / 100.00% | 55,849 / 1,402 / 100.00% |  |
| Patio_Lawn_and_Garden | leave_last_out | 6/21 | 98,613 | 14,019 | 668,776 | 98,613 / 13,333 / 0.00% | 98,613 / 12,826 / 0.00% |  |
| Patio_Lawn_and_Garden | temporal | 5/13 | 89,415 | 14,329 | 269,307 | 52,279 / 10,667 / 35.60% | 28,043 / 9,718 / 46.72% |  |
| Pet_Supplies | user_split | 10/16 | 89,150 | 19,880 | 1,238,089 | 5,000 / 8,436 / 0.00% | 5,000 / 8,533 / 0.00% |  |
| Pet_Supplies | item_split | 10/16 | 89,150 | 19,880 | 1,182,394 | 46,784 / 994 / 100.00% | 68,722 / 1,988 / 100.00% |  |
| Pet_Supplies | leave_last_out | 10/16 | 89,150 | 19,880 | 1,216,185 | 89,150 / 16,106 / 0.00% | 89,150 / 15,361 / 0.00% |  |
| Pet_Supplies | temporal | 8/12 | 88,279 | 18,400 | 584,750 | 47,781 / 11,921 / 19.44% | 34,883 / 12,385 / 36.15% |  |
| Software | user_split | 6/6 | 98,634 | 13,664 | 916,962 | 5,000 / 4,254 / 0.00% | 5,000 / 4,129 / 0.00% |  |
| Software | item_split | 6/6 | 98,634 | 13,664 | 870,039 | 34,962 / 684 / 100.00% | 62,676 / 1,367 / 100.00% |  |
| Software | leave_last_out | 6/6 | 98,634 | 13,664 | 823,831 | 98,634 / 9,800 / 0.17% | 98,634 / 9,400 / 0.27% |  |
| Software | temporal | 2/2 | 76,101 | 14,548 | 232,769 | 29,981 / 5,902 / 22.89% | 14,134 / 3,762 / 24.83% |  |
| Sports_and_Outdoors | user_split | 6/17 | 99,084 | 19,023 | 815,594 | 5,000 / 7,398 / 0.00% | 5,000 / 7,431 / 0.00% |  |
| Sports_and_Outdoors | item_split | 6/17 | 99,084 | 19,023 | 775,040 | 35,955 / 952 / 100.00% | 55,955 / 1,903 / 100.00% |  |
| Sports_and_Outdoors | leave_last_out | 6/17 | 99,084 | 19,023 | 709,121 | 99,084 / 17,601 / 0.00% | 99,084 / 16,816 / 0.00% |  |
| Sports_and_Outdoors | temporal | 5/9 | 61,892 | 18,592 | 234,732 | 29,250 / 10,372 / 24.21% | 20,849 / 11,653 / 48.85% |  |
| Subscription_Boxes | user_split | 2/1 | 572 | 327 | 1,029 | 42 / 28 / 0.00% | 47 / 36 / 0.00% | † Small evaluation set |
| Subscription_Boxes | item_split | 2/1 | 572 | 327 | 1,044 | 56 / 16 / 100.00% | 141 / 31 / 100.00% | † Small evaluation set |
| Subscription_Boxes | leave_last_out | 4/1 | 24 | 86 | 68 | 24 / 23 / 78.26% | 24 / 21 / 95.24% | † Small evaluation set |
| Subscription_Boxes | temporal | 2/1 | 152 | 191 | 133 | 59 / 62 / 77.42% | 42 / 34 / 88.24% | † Small evaluation set |
| Tools_and_Home_Improvement | user_split | 8/26 | 94,834 | 17,307 | 1,067,854 | 5,000 / 8,377 / 0.00% | 5,000 / 8,199 / 0.00% |  |
| Tools_and_Home_Improvement | item_split | 8/26 | 94,834 | 17,307 | 1,006,137 | 45,023 / 866 / 100.00% | 66,710 / 1,731 / 100.00% |  |
| Tools_and_Home_Improvement | leave_last_out | 8/26 | 94,834 | 17,307 | 1,003,805 | 94,834 / 15,754 / 0.00% | 94,834 / 15,151 / 0.00% |  |
| Tools_and_Home_Improvement | temporal | 6/18 | 99,815 | 15,514 | 473,873 | 52,511 / 11,518 / 21.32% | 37,242 / 11,481 / 33.10% |  |
| Toys_and_Games | user_split | 6/22 | 98,065 | 15,609 | 828,331 | 5,000 / 7,090 / 0.00% | 5,000 / 7,031 / 0.00% |  |
| Toys_and_Games | item_split | 6/22 | 98,065 | 15,609 | 787,497 | 34,026 / 781 / 100.00% | 56,767 / 1,561 / 100.00% |  |
| Toys_and_Games | leave_last_out | 6/22 | 98,065 | 15,609 | 725,441 | 98,065 / 14,818 / <0.01% | 98,065 / 14,342 / <0.01% |  |
| Toys_and_Games | temporal | 5/10 | 55,927 | 17,369 | 234,102 | 23,364 / 8,553 / 20.29% | 20,412 / 11,428 / 54.60% |  |
| Video_Games | user_split | 5/7 | 88,081 | 19,227 | 667,965 | 5,000 / 5,574 / 0.00% | 5,000 / 5,551 / 0.00% |  |
| Video_Games | item_split | 5/7 | 88,081 | 19,227 | 635,194 | 30,322 / 962 / 100.00% | 48,790 / 1,923 / 100.00% |  |
| Video_Games | leave_last_out | 5/7 | 88,081 | 19,227 | 577,477 | 88,081 / 15,526 / 0.15% | 88,081 / 15,042 / 0.17% |  |
| Video_Games | temporal | 3/4 | 70,806 | 16,463 | 183,119 | 30,103 / 7,650 / 29.50% | 21,917 / 7,253 / 48.01% |  |

Verification: 1,571 tests passed, 4 skipped; strict Sphinx documentation build passed. The audit verifies reproducibility and measured sizes, not recommendation quality or global threshold optimality. Tiny evaluation sets, especially Subscription Boxes, remain unsuitable for reliable model comparisons.
