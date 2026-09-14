# Translation results

Model `facebook/nllb-200-distilled-600M`, CPU. Real generated sentences from the 18-report evaluation corpus. Regenerate with `python -m eval.translation_eval`.

| Metric | Value |
| --- | ---: |
| Forward translation (min / median / max) | 5.6s / 8.5s / 14.7s |
| Round trip, forward + back (min / median / max) | 9.3s / 15.3s / 24.3s |
| Back-translation similarity to original (min / median) | 0.863 / 0.907 |

**Only Hindi is enabled in the app.** Telugu and Malayalam are measured here for comparison, not built — see app/i18n/translator.py for the latency reasoning.

## Every measured pair

| Language | Forward (s) | Back (s) | Similarity | English | Forward translation | Back-translation |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Hindi | 14.7 | 8.1 | 0.890 | Your report mentions "hepatic steatosis". What it means: Extra fat has built up inside the liver. This is a common finding and is often picked up by chance on a scan. | आपकी रिपोर्ट में "एपेटिक स्टेटोसिस" का उल्लेख है। इसका क्या मतलब है: यकृत के अंदर अतिरिक्त वसा जमा हो गया है। यह एक आम निष्कर्ष है और अक्सर स्कैन पर आकस्मिक रूप से उठाया जाता है। | What this means: extra fat has accumulated inside the liver. This is a common finding and is often taken accidentally on a scan. |
| Hindi | 7.9 | 6.8 | 0.975 | Extra fat has built up inside the liver, which is a common finding and is often picked up by chance on a scan. | यकृत के अंदर अतिरिक्त वसा जमा हो गया है, जो एक आम खोज है और अक्सर स्कैन पर आकस्मिक रूप से उठाया जाता है। | Additional fat has accumulated inside the liver, which is a common finding and is often picked up accidentally on a scan. |
| Hindi | 6.3 | 5.1 | 0.863 | The liver enzyme ALT is higher than usual, which can be a sign that the liver is irritated. | यकृत एंजाइम ALT सामान्य से अधिक है, जो यकृत की जलन का संकेत हो सकता है। | The liver enzyme ALT is higher than normal, which may indicate inflammation of the liver. |
| Telugu | 13.7 | 10.5 | 0.873 | Your report mentions "hepatic steatosis". What it means: Extra fat has built up inside the liver. This is a common finding and is often picked up by chance on a scan. | మీ నివేదికలో "హేపటిక్ స్టీటోసిస్" గురించి చెప్పబడింది. దీని అర్థంః కాలేయం లోపల అదనపు కొవ్వు పెరిగింది. ఇది ఒక సాధారణ ఫలితం మరియు తరచుగా స్కాన్ చేసినప్పుడు యాదృచ్ఛికంగా గుర్తించబడుతుంది. | Your report says about "hepatic stenosis". This means: an increase in excess fat inside the liver. This is a common result and can be detected randomly when scanned frequently. |
| Telugu | 8.7 | 6.6 | 0.933 | Extra fat has built up inside the liver, which is a common finding and is often picked up by chance on a scan. | అదనపు కొవ్వు కాలేయం లోపల సేకరించింది, ఇది ఒక సాధారణ ఆవిష్కరణ మరియు తరచుగా ఒక స్కాన్ వద్ద యాదృచ్ఛికంగా తీయబడుతుంది. | The extra fat is collected inside the liver, which is a common finding and is often taken randomly at a scan. |
| Telugu | 6.9 | 5.2 | 0.936 | The liver enzyme ALT is higher than usual, which can be a sign that the liver is irritated. | కాలేయ ఎంజైమ్ ALT సాధారణం కంటే ఎక్కువగా ఉంటుంది, ఇది కాలేయ చికాకుకు యొక్క సంకేతం కావచ్చు. | Liver enzyme ALT is higher than normal, which may be a sign of liver irritation. |
| Malayalam | 14.6 | 8.1 | 0.907 | Your report mentions "hepatic steatosis". What it means: Extra fat has built up inside the liver. This is a common finding and is often picked up by chance on a scan. | നിങ്ങളുടെ റിപ്പോർട്ടിൽ "ഹെപ്പാറ്റിക് സ്റ്റെറ്റോസിസ്" എന്ന് പരാമർശിക്കുന്നു. അതിനർത്ഥം: കരൾ ഉള്ളിൽ അധിക കൊഴുപ്പ് കൂടിയിട്ടുണ്ട്. ഇത് സാധാരണ കണ്ടെത്തലാണ്. | Your report refers to it as "hepatic stenosis", which means that there is excess fat inside the liver, which is a common finding. |
| Malayalam | 8.5 | 7.0 | 0.922 | Extra fat has built up inside the liver, which is a common finding and is often picked up by chance on a scan. | അധിക കൊഴുപ്പ് കരൾക്കകത്തുണ്ടായിട്ടുണ്ട്. ഇത് ഒരു സാധാരണ കണ്ടെത്തലാണ്. ഇത് പലപ്പോഴും ഒരു സ്കാനിൽ ആകസ്മികമായി കണ്ടെത്തിയിട്ടുണ്ട്. | There's excess fat in the liver, which is a common finding, and it's often found accidentally on a scan. |
| Malayalam | 5.6 | 3.7 | 0.868 | The liver enzyme ALT is higher than usual, which can be a sign that the liver is irritated. | കരൾ എൻസൈം ALT സാധാരണയേക്കാൾ കൂടുതലാണ്, ഇത് കരൾ തകരാറിലായതിന്റെ സൂചനയാകാം. | The liver enzyme ALT is higher than normal, which may indicate liver failure. |

## Why a high similarity score is not proof of a correct translation

Round-tripping a specialised clinical term through a general-purpose translation model can corrupt it in a way that still round-trips to something plausible. Look for any row above where the English mentions "steatosis" (fat in the liver) and the back-translation says "stenosis" (a narrowing — a different condition) instead: if present, it shows the back-translation score alone would have missed a clinically meaningful error, because the sentence *shape* survived the round trip even though a specific medical fact did not. This is exactly why this check is offline and advisory, not a live gate — the same reasoning as the semantic grounding check in step 6 (eval/semantic_grounding_results.md).
