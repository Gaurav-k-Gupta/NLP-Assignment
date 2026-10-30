# Curated qualitative review

Five distinct held-out test examples. These were selected to show one close match and several observed failure types. This is a purposive case review, not a statistical estimate of error frequency.

## Example 1: `225f2330a6477e05fd3cebb31d3c87243cd40eb6`

**Assessment:** Strong case. Generated text closely reproduces the reference and preserves its sentence breaks. This is an unusually easy, page-description example; it does not represent typical news summarization quality.

**Article excerpt:** April 8, 2015 . After a report on headlines concerning the U.S. and Cuba, CNN Student News gives some perspective on California's historic drought: Its effects are becoming more likely to ripple across the nation. There's been a flu outbreak in Chicago -- but not one that affects humans. And a proposed tunnel would connect Denmark and Germany via an undersea road and rail line. On this page you will find today's show Transcript and a place for you to request to be on the CNN Student News Roll Call. TRANSCRIPT . Click here to access the transcript of today's CNN Student News program. Please note that there may be a delay between the time when the video is available and when the transcript is published. CNN Student News is created by a team of journalists who consider the Common Core State Standards, national standards in different subject 

**Reference:**

This page includes the show Transcript .
Use the Transcript to help students with reading comprehension and vocabulary .
At the bottom of the page, comment for a chance to be mentioned on CNN Student News.  You must be a teacher or a student age 13 or older to request a mention on the CNN Student News Roll Call.

**Generated summary:**

this page includes the show transcript.
use the transcript to help students with reading comprehension and vocabulary.
at the bottom of the page, comment for a chance to be mentioned on cnn student news. you must be a teacher or a student age 13 or older to request a mention on the cnn student news roll call.


## Example 2: `6555c15003defec00c965b3851e4da818264afe7`

**Assessment:** Severe repetition and unsupported wording. It partly mentions the $1,000 amount but corrupts the claim, repeats “more likely,” and invents a “new girlfriend.”

**Article excerpt:** Promposals are the big trend in high school proms as teens think up progressively more outlandish ways to ask each other to the biggest dance of the year. They are so big, in fact, that students this year will likely spend of a third of their prom budgets on the elaborate new ritual, according to a new survey. With budgets for outfits, flowers, transportation, food and other sundries at around $1,000 for the average American high schooler, that means most intend to spend $324 on the once-simple query: Will you go to prom with me? Pricey promposals: Students this year will likely spend of a third of their prom budgets on the elaborate new ritual of promposals, according to a survey of 3,000 people conducted by Visa . That's according to a survey from Visa, which asked 3,000 people aged 18 or older what they expected to shell out for the pr

**Reference:**

Students spend $1,000 a year on prom on average, according to survey from Visa .

**Generated summary:**

survey found that the average american people are more likely to be more likely to be more likely to be more likely to be more likely.
the average of $ 1, 000 is the most popular in the u. s.
the average - year - old student is a new girlfriend.


## Example 3: `5be3918cf2650cf946c66ac5a3b19ab38d6be66a`

**Assessment:** Factual contradiction. The reference says Manchester United lost 3–0 to Everton. The output claims a 2–0 win over West Ham, then inconsistently says United were beaten by Everton.

**Article excerpt:** Louis van Gaal accused Manchester United’s players of lacking desire after a 3-0 defeat at Everton put their top four hopes back in the balance. The irritation that United’s manager felt following the worst day he has experienced in the Barclays Premier League was obvious and he did not attempt to make any excuses for a performance that was toothless and meek from the outset. Goals from James McCarthy, John Stones and substitute Kevin Mirallas did not flatter Everton and Van Gaal claimed the difference between the two sides came in the attitude and approach they adopted, as United could not match Everton’s commitment. Wayne Rooney and Radamel Falcao look dejected as they wait to kick off after the third goal . Manchester United boss Louis van Gaal said that his players could not match Everton's motivation . West Brom (home) - May 2 . Crys

**Reference:**

Manchester United lost 3-0 by Everton at Goodison Park on Sunday .
Louis van Gaal said United's motivation did not match Everton's .
The United boss could tell the attitude was not right in the warm-up .

**Generated summary:**

manchester united beat west ham 2 - 0 in the premier league on saturday.
louis van gaal ' s side were beaten by everton in the premier league.
the dutchman was also told he was ' a good player '
the manchester united boss said he was ' very happy ' to play.
the united boss said he was ' very happy ' to play for the club.


## Example 4: `bd3b1f690ea5b237df0c08ebbc2ed37c0be2738b`

**Assessment:** Decoder repetition loop. It repeats “dr.” throughout and conveys none of the article’s claims about Alice Dreger or the abstinence-only lesson.

**Article excerpt:** Worried: Alice Dreger attended the class at East Lansing High School after her son asked her to sit in on it so she could hear what was going on first hand . An outraged mom live tweeted her son's sex education class after becoming outraged by its abstinence only stance. Author Alice Dreger took to Twitter to express her horror at what she heard at East Lansing High School, in Michigan, when she sat in on a lesson earlier this week. Ms Dreger, a respected professor, began furiously tweeting out the contents of the lesson - which was filled with horror stories about what happened to people who have pre-marital sex. According to the 45 tweets sent by Ms Dreger during the lesson, stories included people who lost all their friends when they became pregnant, people who fell into comas for 11 years after taking drugs and high school dropouts. '

**Reference:**

Author Alice Dreger sent 45 angry tweets during her son's sex ed lesson .
Could not believe Michigan school was teaching abstinence only classes .
'The whole lesson here is "sex is part of a terrible lifestyle",' she fumed .
Principal Coby Fletcher denied the school has an 'abstinence only' policy .

**Generated summary:**

dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr. dr.


## Example 5: `13ec87a8f0fb2021b2fc9b927d75fd5a0acc08c4`

**Assessment:** Degenerate token loop. It repeats “al” and omits the stadium, World Cup, capacity, date, and cooling-system facts.

**Article excerpt:** Qatar have unveiled the design for their fifth World Cup stadium, a 40,000-seater ground built in 'the gateway to the desert' and featuring air conditioning for the comfort of players and fans. The Al Rayyan Stadium is inspired by sand dunes and its design has been produced by UK-based architecture firms Ramboll and Pattern. Fans in the stands will sit in a comfortable 24-28 degrees Celsius, while paths and walkways leading to the stadium will also be cooled. Qatar have unveiled the design for their fifth World Cup stadium, the 40,000-seater Al Rayyan Stadium . The Al Rayyan Stadium is inspired by sand dunes and its design has been produced by UK-based firms . The stadium is expected to host matches up to the quarter-final stage of the World Cup in Qatar in 2022 . Al Rayyan will hold matches up to the quarter-final stage of the 2022 tourn

**Reference:**

Qatar's Supreme Committee unveil new 40,000-seat stadium for World Cup .
The Al Rayyan Stadium is fifth 2022 venue and will be completed in 2019 .
Stadium features 'cooling technology' for fans' and players' comfort .

**Generated summary:**

the al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al - al
