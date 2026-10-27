# Hand check v1: 10 questions (AN-001, acceptance criterion 3)

**Goal:** for every answer, each cited source really says what the answer claims.
**Setup:** `uv run ai-act ask --quotes "<question>"`, model `claude-sonnet-5-5`, effort `medium`, k = 5, prompt `v1`.
Run on 2026-10-26. Total cost about $0.11.

**How to check:** open the answer below, compare each claim with the quoted passage under its source (or click the EUR-Lex link), then fill in the last two columns.
The "Pre-check" column is Claude's own first pass. The verdict is yours.

| # | Question | Sources cited | Pre-check (Claude) | Supported? (yes / no) | Note |
|---|---|---|---|---|---|
| 1 | Does a chatbot have to tell users it is an AI? | Art. 50(1), Recital 132 | Citations match. ⚠️ Retrieved but uncited: the last sentence names Art. 50(2), which was among the 5 chunks sent but carries no [number]. | yes | Art. 50(2) retrieved but not cited. |
| 2 | Which AI practices are prohibited? | Art. 5(1), point (a), Art. 5(1), point (b), Art. 5(1), point (d), Art. 5(1), point (e), Art. 5(1), point (f) | Citations match. ⚠️ Incomplete: only 5 of 8 banned practices were retrieved; the answer says so. | yes | Incomplete (5 of 8 practices); the answer says so. |
| 3 | What makes an AI system high-risk? | Art. 6(1), Annex III, point 2, Art. 15(1), Art. 9(6) | Citations match. ⚠️ Incomplete: only 1 Annex III area retrieved; the answer says so. | yes | Art. 6(3) exceptions not retrieved; the answer says it can't assess them. |
| 4 | What are the maximum fines for breaking the prohibitions? | Art. 99(3), Art. 100(2), Art. 99(4) | Citations match (EUR 35M / 7 %, EUR 1.5M for EU bodies, EUR 15M / 3 %). ⚠️ Incomplete: for SMEs and start-ups the *lower* of the two amounts applies (Art. 99(6)), which wasn't retrieved, so the answer doesn't mention it. | yes | Missing: SME rule in Art. 99(6) (not retrieved). |
| 5 | Do deepfakes have to be labelled? | Art. 50(4), Recital 134 | Citations match, incl. the text rule in Art. 50(4) second subparagraph. | yes | Missing: the text rule leaves out the law-enforcement exception, though it is in the quote. |
| 6 | What obligations do providers of general-purpose AI models have? | Art. 53(1), Art. 53(3), Art. 54(2), Art. 53(2), Art. 54(6) | Citations match. One uncited conclusion (points (c), (d) not exempt), which follows from Art. 53(2). | yes | Uncited conclusion follows from Art. 53(2). |
| 7 | Who counts as a provider under the AI Act? | Art. 3, point (3) | Citation matches the definition word for word. | yes | |
| 8 | Is emotion recognition in the workplace allowed? | Recital 44, Art. 3, point (39), Recital 18, Art. 50(3) | ⚠️ Known search miss: relies on Recital 44, not the ban itself (Art. 5(1)(f)). The answer says so. ⚠️ Check [4]: "Outside the workplace and education context" is Claude's own framing; Art. 50(3) applies to deployers of emotion recognition in general and doesn't mention the workplace. | yes | The workplace framing is a fair inference (the workplace case is banned, so Art. 50(3) applies outside it), not in the quote. Leaves out Art. 50(3)'s law-enforcement exception. |
| 9 | When do the rules for high-risk AI systems start to apply? | Art. 111(2), Art. 6(5) | ⚠️ Known search miss: Art. 113 not retrieved. The answer says it can't give the date. Cited dates correct. ⚠️ Outside knowledge: "Article 113, which would set it" isn't in any chunk sent (Art. 111(2) only mentions Art. 113(3), point (a), for Article 5). | yes | Outside knowledge: "Article 113, which would set it". |
| 10 | What is the VAT rate in Germany? | none | Correctly declines, no sources. | yes | Correctly declines. |

**Summary:** 10 of 10 answers fully supported by their citations (verdicts by the project owner, 2026-10-27).

**How the verdicts were decided:** "yes" means every claim with a [number] matches its quote. Missing information is a note, not a "no". A claim the quote contradicts or doesn't contain would be a "no". Two LLM-judge passes were used as a pre-check; without that rule they disagreed on Q4, which is why the automatic LLM judge (AN-004) needs it in its prompt.

**Known limits found:**
- **Search misses (fix in AN-003):** questions 8 and 9 don't retrieve the key article.
- **Incomplete lists:** questions 2 and 3 only get part of the list, because only 5 chunks are sent.
- **Retrieved but uncited:** question 1 names Art. 50(2), which was sent but not cited.
- **Outside knowledge:** question 9 says what Art. 113 does, though Art. 113 wasn't sent. Question 8 adds its own framing to a cited source.
- AN-002 will count these two failure types separately: "retrieved but uncited" means the source was there, "outside knowledge" means the model added something the excerpts don't contain.

## Answers

### 1. Does a chatbot have to tell users it is an AI?

<details><summary>Full answer with sources and quotes ($0.0120)</summary>

```text
Yes, in general. Providers must ensure that AI systems intended to interact directly with natural persons are designed and developed so that the people concerned are informed they are interacting with an AI system. [1] A chatbot falls squarely within that description.

**Exceptions and conditions:**
- **Obvious from context:** The duty doesn't apply if it is obvious from the point of view of a reasonably well-informed, observant and circumspect person, taking into account the circumstances and the context of use. [1]
- **Law enforcement:** The obligation doesn't apply to AI systems authorised by law to detect, prevent, investigate or prosecute criminal offences, subject to appropriate safeguards for third parties' rights and freedoms, unless those systems are available to the public to report a criminal offence. [1]

**Other points:**
- The obligation sits with the provider and is built into design and development, so the disclosure should be part of the product itself.
- When implementing it, the characteristics of vulnerable groups due to their age or disability should be taken into account to the extent the system is intended to interact with those groups, and the information should be provided in accessible formats for persons with disabilities. [2]
- This transparency requirement applies irrespective of whether the system qualifies as high-risk, and it is without prejudice to high-risk requirements. [2]

If your chatbot also generates synthetic text, audio, image or video, there are separate marking obligations under Art. 50(2).

Sources:
  [1] Art. 50(1)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_50
      "1. Providers shall ensure that AI systems intended to interact directly with natural persons are designed and developed in such a way that the natural persons concerned are informed that they are interacting with an AI system, unless this is obvious from the point of view of a natural person who is reasonably well-informed, observant and circumspect, taking into account the circumstances and the context of use. This obligation shall not apply to AI systems authorised by law to detect, prevent, investigate or prosecute criminal offences, subject to appropriate safeguards for the rights and freedoms of third parties, unless those systems are available for the public to report a criminal offence."
  [2] Recital 132  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#rct_132
      "(132) Certain AI systems intended to interact with natural persons or to generate content may pose specific risks of impersonation or deception irrespective of whether they qualify as high-risk or not. In certain circumstances, the use of these systems should therefore be subject to specific transparency obligations without prejudice to the requirements and obligations for high-risk AI systems and subject to targeted exceptions to take into account the special need of law enforcement. In particular, natural persons should be notified that they are interacting with an AI system, unless this is obvious from the point of view of a natural person who is reasonably well-informed, observant and circumspect taking into account the circumstances and the context of use. When implementing that obligation, the characteristics of natural persons belonging to vulnerable groups due to their age or disability should be taken into account to the extent the AI system is intended to interact with those groups as well. Moreover, natural persons should be notified when they are exposed to AI systems that, by processing their biometric data, can identify or infer the emotions or intentions of those persons or assign them to specific categories. Such specific categories can relate to aspects such as sex, age, hair colour, eye colour, tattoos, personal traits, ethnic origin, personal preferences and interests. Such information and notifications should be provided in accessible formats for persons with disabilities."

claude-sonnet-5-5 · 3246 tokens in, 547 out · cost $0.0120
```

</details>

### 2. Which AI practices are prohibited?

<details><summary>Full answer with sources and quotes ($0.0137)</summary>

```text
The excerpts cover five of the prohibited practices under Article 5(1), points (a), (b), (d), (e) and (f). They don't include points (c), (g), (h) or any later points, so this list may not be complete.

1. **Subliminal or manipulative techniques (point a).** This covers AI systems that deploy subliminal techniques beyond a person's consciousness, or purposefully manipulative or deceptive techniques. The objective or effect must be materially distorting behaviour by appreciably impairing the ability to make an informed decision. That must cause a decision the person would not otherwise have taken, in a manner that causes or is reasonably likely to cause significant harm. [1]

2. **Exploiting vulnerabilities (point b).** This covers AI systems that exploit vulnerabilities of a person or group due to age, disability, or a specific social or economic situation. The objective or effect must be materially distorting behaviour in a manner that causes or is reasonably likely to cause significant harm. [2]

3. **Criminal-offence risk assessment based solely on profiling (point d).** This covers AI systems that assess or predict the risk of a person committing a criminal offence based solely on profiling or on assessing personality traits and characteristics. [3] The exception: the prohibition does not apply to AI systems that support human assessment of a person's involvement in a criminal activity, where that assessment is already based on objective and verifiable facts directly linked to a criminal activity. [3]

4. **Untargeted facial image scraping (point e).** This covers AI systems that create or expand facial recognition databases through untargeted scraping of facial images from the internet or CCTV footage. [4]

5. **Emotion inference in workplaces and education (point f).** This covers AI systems that infer the emotions of a natural person in the areas of workplace and education institutions. [5] The exception: it does not apply where the system is intended to be put in place or into the market for medical or safety reasons. [5]

For points (a) and (b), the prohibition applies to placing on the market, putting into service, or use [1] [2]. For points (d), (e) and (f), the wording is "putting into service for this specific purpose" [5] [4] [3] rather than just putting into service.

Sources:
  [1] Art. 5(1), point (a)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_5
      "1. The following AI practices shall be prohibited: (a) the placing on the market, the putting into service or the use of an AI system that deploys subliminal techniques beyond a person’s consciousness or purposefully manipulative or deceptive techniques, with the objective, or the effect of materially distorting the behaviour of a person or a group of persons by appreciably impairing their ability to make an informed decision, thereby causing them to take a decision that they would not have otherwise taken in a manner that causes or is reasonably likely to cause that person, another person or group of persons significant harm;"
  [2] Art. 5(1), point (b)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_5
      "1. The following AI practices shall be prohibited: (b) the placing on the market, the putting into service or the use of an AI system that exploits any of the vulnerabilities of a natural person or a specific group of persons due to their age, disability or a specific social or economic situation, with the objective, or the effect, of materially distorting the behaviour of that person or a person belonging to that group in a manner that causes or is reasonably likely to cause that person or another person significant harm;"
  [3] Art. 5(1), point (d)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_5
      "1. The following AI practices shall be prohibited: (d) the placing on the market, the putting into service for this specific purpose, or the use of an AI system for making risk assessments of natural persons in order to assess or predict the risk of a natural person committing a criminal offence, based solely on the profiling of a natural person or on assessing their personality traits and characteristics; this prohibition shall not apply to AI systems used to support the human assessment of the involvement of a person in a criminal activity, which is already based on objective and verifiable facts directly linked to a criminal activity;"
  [4] Art. 5(1), point (e)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_5
      "1. The following AI practices shall be prohibited: (e) the placing on the market, the putting into service for this specific purpose, or the use of AI systems that create or expand facial recognition databases through the untargeted scraping of facial images from the internet or CCTV footage;"
  [5] Art. 5(1), point (f)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_5
      "1. The following AI practices shall be prohibited: (f) the placing on the market, the putting into service for this specific purpose, or the use of AI systems to infer emotions of a natural person in the areas of workplace and education institutions, except where the use of the AI system is intended to be put in place or into the market for medical or safety reasons;"

claude-sonnet-5-5 · 2514 tokens in, 871 out · cost $0.0137
```

</details>

### 3. What makes an AI system high-risk?

<details><summary>Full answer with sources and quotes ($0.0109)</summary>

```text
The excerpts give two routes to high-risk status, but they cover only part of the classification rules.

**Route 1: Safety component or product under Annex I legislation (Art. 6(1)).** An AI system is high-risk when both conditions are met: (a) it is intended to be used as a safety component of a product, or is itself a product, covered by the Union harmonisation legislation listed in Annex I; and (b) that product (or the AI system itself as a product) must undergo a third-party conformity assessment before being placed on the market or put into service under that legislation. [1] This applies whether or not the AI system is placed on the market or put into service independently of the product. [1]

**Route 2: Listed use areas in Annex III (Art. 6(2)).** AI systems listed in the Annex III areas are high-risk. [2] The excerpt shows only one area, critical infrastructure: AI systems intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic, or in the supply of water, gas, heating or electricity. [2]

**Limits of these excerpts:** They don't include the other Annex III areas, the full text of Art. 6(2)–(3), or the contents of Annex I, so I can't say what else qualifies or whether any exceptions apply.

Once a system is high-risk, requirements follow, such as appropriate accuracy, robustness and cybersecurity throughout its lifecycle [3] and testing to identify the most appropriate and targeted risk management measures [4].

Sources:
  [1] Art. 6(1)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_6
      "1. Irrespective of whether an AI system is placed on the market or put into service independently of the products referred to in points (a) and (b), that AI system shall be considered to be high-risk where both of the following conditions are fulfilled: (a) the AI system is intended to be used as a safety component of a product, or the AI system is itself a product, covered by the Union harmonisation legislation listed in Annex I; (b) the product whose safety component pursuant to point (a) is the AI system, or the AI system itself as a product, is required to undergo a third-party conformity assessment, with a view to the placing on the market or the putting into service of that product pursuant to the Union harmonisation legislation listed in Annex I."
  [2] Annex III, point 2  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#anx_III
      "High-risk AI systems pursuant to Article 6(2) are the AI systems listed in any of the following areas: 2. Critical infrastructure: AI systems intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic, or in the supply of water, gas, heating or electricity."
  [3] Art. 15(1)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_15
      "1. High-risk AI systems shall be designed and developed in such a way that they achieve an appropriate level of accuracy, robustness, and cybersecurity, and that they perform consistently in those respects throughout their lifecycle."
  [4] Art. 9(6)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_9
      "6. High-risk AI systems shall be tested for the purpose of identifying the most appropriate and targeted risk management measures. Testing shall ensure that high-risk AI systems perform consistently for their intended purpose and that they are in compliance with the requirements set out in this Section."

claude-sonnet-5-5 · 2556 tokens in, 582 out · cost $0.0109
```

</details>

### 4. What are the maximum fines for breaking the prohibitions?

<details><summary>Full answer with sources and quotes ($0.0083)</summary>

```text
Maximum fines depend on who breaks the Article 5 prohibitions:

- **Private operators (e.g. companies):** Non-compliance with the prohibited AI practices in Article 5 is subject to administrative fines of up to EUR 35,000,000 or, if the offender is an undertaking, up to 7% of its total worldwide annual turnover for the preceding financial year, whichever is higher. [1]

- **EU institutions, bodies, offices and agencies:** Non-compliance with the Article 5 prohibitions is subject to administrative fines of up to EUR 1,500,000. [2]

For comparison, the excerpts show that other breaches are capped lower. For example, breaches of certain provider, importer, distributor and deployer obligations, among others, carry fines of up to EUR 15,000,000 or 3% of worldwide annual turnover, whichever is higher. [3]

Sources:
  [1] Art. 99(3)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_99
      "3. Non-compliance with the prohibition of the AI practices referred to in Article 5 shall be subject to administrative fines of up to EUR 35 000 000 or, if the offender is an undertaking, up to 7 % of its total worldwide annual turnover for the preceding financial year, whichever is higher."
  [2] Art. 100(2)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_100
      "2. Non-compliance with the prohibition of the AI practices referred to in Article 5 shall be subject to administrative fines of up to EUR 1 500 000."
  [3] Art. 99(4)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_99
      "4. Non-compliance with any of the following provisions related to operators or notified bodies, other than those laid down in Articles 5, shall be subject to administrative fines of up to EUR 15 000 000 or, if the offender is an undertaking, up to 3 % of its total worldwide annual turnover for the preceding financial year, whichever is higher: (a) obligations of providers pursuant to Article 16; (b) obligations of authorised representatives pursuant to Article 22; (c) obligations of importers pursuant to Article 23; (d) obligations of distributors pursuant to Article 24; (e) obligations of deployers pursuant to Article 26; (f) requirements and obligations of notified bodies pursuant to Article 31, Article 33(1), (3) and (4) or Article 34; (g) transparency obligations for providers and deployers pursuant to Article 50."

claude-sonnet-5-5 · 2432 tokens in, 342 out · cost $0.0083
```

</details>

### 5. Do deepfakes have to be labelled?

<details><summary>Full answer with sources and quotes ($0.0120)</summary>

```text
Yes. Deployers of an AI system that generates or manipulates image, audio or video content constituting a deep fake must disclose that the content has been artificially generated or manipulated. [1] The recitals describe deep fakes as content that appreciably resembles existing persons, objects, places, entities or events and would falsely appear to a person to be authentic or truthful [2], and say the disclosure should be clear and distinguishable.

**Exceptions and limits:**
- **Law enforcement use:** The obligation does not apply where the use is authorised by law to detect, prevent, investigate or prosecute criminal offences. [1]
- **Artistic, creative, satirical or fictional works:** Where the content forms part of an evidently artistic, creative, satirical, fictional or analogous work or programme, the obligation is limited to disclosing the existence of the generated or manipulated content in an appropriate manner that does not hamper the display or enjoyment of the work. [1]

**AI-generated text:** A similar duty applies to text. Deployers of an AI system that generates or manipulates text published with the purpose of informing the public on matters of public interest must disclose that it was artificially generated or manipulated. [1] This does not apply where the AI-generated content has undergone a process of human review or editorial control and a natural or legal person holds editorial responsibility for the publication. [1]

The excerpts place this obligation on deployers. The recital also refers to technical solutions employed by the providers of the AI system [2], but the excerpts don't detail what those involve.

Sources:
  [1] Art. 50(4)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_50
      "4. Deployers of an AI system that generates or manipulates image, audio or video content constituting a deep fake, shall disclose that the content has been artificially generated or manipulated. This obligation shall not apply where the use is authorised by law to detect, prevent, investigate or prosecute criminal offence. Where the content forms part of an evidently artistic, creative, satirical, fictional or analogous work or programme, the transparency obligations set out in this paragraph are limited to disclosure of the existence of such generated or manipulated content in an appropriate manner that does not hamper the display or enjoyment of the work. Deployers of an AI system that generates or manipulates text which is published with the purpose of informing the public on matters of public interest shall disclose that the text has been artificially generated or manipulated. This obligation shall not apply where the use is authorised by law to detect, prevent, investigate or prosecute criminal offences or where the AI-generated content has undergone a process of human review or editorial control and where a natural or legal person holds editorial responsibility for the publication of the content."
  [2] Recital 134  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#rct_134
      "(134) Further to the technical solutions employed by the providers of the AI system, deployers who use an AI system to generate or manipulate image, audio or video content that appreciably resembles existing persons, objects, places, entities or events and would falsely appear to a person to be authentic or truthful (deep fakes), should also clearly and distinguishably disclose that the content has been artificially created or manipulated by labelling the AI output accordingly and disclosing its artificial origin. Compliance with this transparency obligation should not be interpreted as indicating that the use of the AI system or its output impedes the right to freedom of expression and the right to freedom of the arts and sciences guaranteed in the Charter, in particular where the content is part of an evidently creative, satirical, artistic, fictional or analogous work or programme, subject to appropriate safeguards for the rights and freedoms of third parties. In those cases, the transparency obligation for deep fakes set out in this Regulation is limited to disclosure of the existence of such generated or manipulated content in an appropriate manner that does not hamper the display or enjoyment of the work, including its normal exploitation and use, while maintaining the utility and quality of the work. In addition, it is also appropriate to envisage a similar disclosure obligation in relation to AI-generated or manipulated text to the extent it is published with the purpose of informing the public on matters of public interest unless the AI-generated content has undergone a process of human review or editorial control and a natural or legal person holds editorial responsibility for the publication of the content."

claude-sonnet-5-5 · 2951 tokens in, 612 out · cost $0.0120
```

</details>

### 6. What obligations do providers of general-purpose AI models have?

<details><summary>Full answer with sources and quotes ($0.0146)</summary>

```text
Providers of general-purpose AI models have four core obligations under Article 53(1), plus a duty to cooperate with authorities.

**Core obligations (Art. 53(1)):**
- **Technical documentation:** Draw up and keep up-to-date the technical documentation of the model, including its training and testing process and the results of its evaluation, containing at a minimum the information in Annex XI, to provide upon request to the AI Office and national competent authorities. [1]
- **Downstream provider information:** Draw up, keep up-to-date and make available information and documentation to providers of AI systems who intend to integrate the model. This must enable them to understand the model's capabilities and limitations and to comply with their own obligations, and contain at least the elements in Annex XII. Intellectual property rights and confidential business information or trade secrets must still be observed and protected. [1]
- **Copyright policy:** Put in place a policy to comply with Union copyright law, in particular to identify and comply with, including through state-of-the-art technologies, a reservation of rights expressed under Article 4(3) of Directive (EU) 2019/790. [1]
- **Training content summary:** Draw up and publicly release a sufficiently detailed summary of the content used for training, following a template provided by the AI Office. [1]

**Cooperation (Art. 53(3)):** Providers must cooperate as necessary with the Commission and national competent authorities in exercising their competences and powers under the Regulation. [2]

**Authorised representative (Art. 54):** The excerpts also refer to an authorised representative. The provider must enable its authorised representative to perform the tasks specified in the mandate. [3] The excerpts don't set out when a representative must be appointed.

**Open-source exception:**
- The documentation obligations in points (a) and (b) don't apply to providers of models released under a free and open-source licence that allows access, usage, modification and distribution, and whose parameters (including weights), architecture information and usage information are made publicly available. [4]
- The Article 54 obligation likewise doesn't apply to such open-source providers, unless the models present systemic risks. [5]
- The Article 53(2) exception does not apply to general-purpose AI models with systemic risks. [4]
- Points (c) and (d), the copyright policy and training summary, are not covered by the exception in Art. 53(2).

The excerpts don't cover the additional obligations for models with systemic risks.

Sources:
  [1] Art. 53(1)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_53
      "1. Providers of general-purpose AI models shall: (a) draw up and keep up-to-date the technical documentation of the model, including its training and testing process and the results of its evaluation, which shall contain, at a minimum, the information set out in Annex XI for the purpose of providing it, upon request, to the AI Office and the national competent authorities; (b) draw up, keep up-to-date and make available information and documentation to providers of AI systems who intend to integrate the general-purpose AI model into their AI systems. Without prejudice to the need to observe and protect intellectual property rights and confidential business information or trade secrets in accordance with Union and national law, the information and documentation shall: (i) enable providers of AI systems to have a good understanding of the capabilities and limitations of the general-purpose AI model and to comply with their obligations pursuant to this Regulation; and (ii) contain, at a minimum, the elements set out in Annex XII; (c) put in place a policy to comply with Union law on copyright and related rights, and in particular to identify and comply with, including through state-of-the-art technologies, a reservation of rights expressed pursuant to Article 4(3) of Directive (EU) 2019/790; (d) draw up and make publicly available a sufficiently detailed summary about the content used for training of the general-purpose AI model, according to a template provided by the AI Office."
  [2] Art. 53(3)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_53
      "3. Providers of general-purpose AI models shall cooperate as necessary with the Commission and the national competent authorities in the exercise of their competences and powers pursuant to this Regulation."
  [3] Art. 54(2)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_54
      "2. The provider shall enable its authorised representative to perform the tasks specified in the mandate received from the provider."
  [4] Art. 53(2)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_53
      "2. The obligations set out in paragraph 1, points (a) and (b), shall not apply to providers of AI models that are released under a free and open-source licence that allows for the access, usage, modification, and distribution of the model, and whose parameters, including the weights, the information on the model architecture, and the information on model usage, are made publicly available. This exception shall not apply to general-purpose AI models with systemic risks."
  [5] Art. 54(6)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_54
      "6. The obligation set out in this Article shall not apply to providers of general-purpose AI models that are released under a free and open-source licence that allows for the access, usage, modification, and distribution of the model, and whose parameters, including the weights, the information on the model architecture, and the information on model usage, are made publicly available, unless the general-purpose AI models present systemic risks."

claude-sonnet-5-5 · 2629 tokens in, 936 out · cost $0.0146
```

</details>

### 7. Who counts as a provider under the AI Act?

<details><summary>Full answer with sources and quotes ($0.0075)</summary>

```text
Under the AI Act, a provider is a natural or legal person, public authority, agency or other body that develops an AI system or a general-purpose AI model, or that has an AI system or a general-purpose AI model developed, and places it on the market or puts the AI system into service under its own name or trademark, whether for payment or free of charge. [1]

Key points from the definition:

- **Who can be a provider:** individuals, companies, public authorities, agencies, or other bodies.
- **Two routes:** you either develop the system or model yourself, or you commission its development from someone else.
- **Own name or trademark:** in both cases, you place it on the market or put the AI system into service under your own name or trademark.
- **Payment is irrelevant:** it applies whether the system or model is offered for payment or free of charge.

The excerpts provided only cover the definition itself. They don't address other situations, such as when a downstream party might become a provider.

Sources:
  [1] Art. 3, point (3)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_3
      "For the purposes of this Regulation, the following definitions apply: (3) ‘provider’ means a natural or legal person, public authority, agency or other body that develops an AI system or a general-purpose AI model or that has an AI system or a general-purpose AI model developed and places it on the market or puts the AI system into service under its own name or trademark, whether for payment or free of charge;"

claude-sonnet-5-5 · 2190 tokens in, 312 out · cost $0.0075
```

</details>

### 8. Is emotion recognition in the workplace allowed?

<details><summary>Full answer with sources and quotes ($0.0123)</summary>

```text
**Generally no.** The excerpts say that the placing on the market, the putting into service, or the use of AI systems intended to be used to detect the emotional state of individuals in situations related to the workplace and education should be prohibited. [1] The stated reasons are the imbalance of power in the context of work or education, combined with the intrusive nature of these systems [1], along with concerns about reliability and discriminatory outcomes.

**Exception:** The prohibition should not cover AI systems placed on the market strictly for medical or safety reasons, such as systems intended for therapeutical use. [1]

**What counts as emotion recognition:**
- An 'emotion recognition system' is an AI system for the purpose of identifying or inferring emotions or intentions of natural persons on the basis of their biometric data. [2]
- It does not include physical states, such as pain or fatigue, for example systems detecting fatigue in professional pilots or drivers to prevent accidents. [3]
- It also does not include the mere detection of readily apparent expressions, gestures or movements, such as a frown, a smile, or a raised voice, unless they are used for identifying or inferring emotions. [3]

**Where the text is limited:** The excerpts come from a recital, which explains the prohibition, and not from the operative prohibition article itself. They don't set out its exact wording or conditions. Outside the workplace and education context, deployers of an emotion recognition system must inform the natural persons exposed to it of its operation and process personal data in accordance with the applicable data protection laws. [4]

Sources:
  [1] Recital 44  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#rct_44
      "(44) There are serious concerns about the scientific basis of AI systems aiming to identify or infer emotions, particularly as expression of emotions vary considerably across cultures and situations, and even within a single individual. Among the key shortcomings of such systems are the limited reliability, the lack of specificity and the limited generalisability. Therefore, AI systems identifying or inferring emotions or intentions of natural persons on the basis of their biometric data may lead to discriminatory outcomes and can be intrusive to the rights and freedoms of the concerned persons. Considering the imbalance of power in the context of work or education, combined with the intrusive nature of these systems, such systems could lead to detrimental or unfavourable treatment of certain natural persons or whole groups thereof. Therefore, the placing on the market, the putting into service, or the use of AI systems intended to be used to detect the emotional state of individuals in situations related to the workplace and education should be prohibited. That prohibition should not cover AI systems placed on the market strictly for medical or safety reasons, such as systems intended for therapeutical use."
  [2] Art. 3, point (39)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_3
      "For the purposes of this Regulation, the following definitions apply: (39) ‘emotion recognition system’ means an AI system for the purpose of identifying or inferring emotions or intentions of natural persons on the basis of their biometric data;"
  [3] Recital 18  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#rct_18
      "(18) The notion of ‘emotion recognition system’ referred to in this Regulation should be defined as an AI system for the purpose of identifying or inferring emotions or intentions of natural persons on the basis of their biometric data. The notion refers to emotions or intentions such as happiness, sadness, anger, surprise, disgust, embarrassment, excitement, shame, contempt, satisfaction and amusement. It does not include physical states, such as pain or fatigue, including, for example, systems used in detecting the state of fatigue of professional pilots or drivers for the purpose of preventing accidents. This does also not include the mere detection of readily apparent expressions, gestures or movements, unless they are used for identifying or inferring emotions. Those expressions can be basic facial expressions, such as a frown or a smile, or gestures such as the movement of hands, arms or head, or characteristics of a person’s voice, such as a raised voice or whispering."
  [4] Art. 50(3)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_50
      "3. Deployers of an emotion recognition system or a biometric categorisation system shall inform the natural persons exposed thereto of the operation of the system, and shall process the personal data in accordance with Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive (EU) 2016/680, as applicable. This obligation shall not apply to AI systems used for biometric categorisation and emotion recognition, which are permitted by law to detect, prevent or investigate criminal offences, subject to appropriate safeguards for the rights and freedoms of third parties, and in accordance with Union law."

claude-sonnet-5-5 · 3086 tokens in, 614 out · cost $0.0123
```

</details>

### 9. When do the rules for high-risk AI systems start to apply?

<details><summary>Full answer with sources and quotes ($0.0090)</summary>

```text
The excerpts don't state a general start date for the high-risk rules, so I can't give you that date from them. Article 113, which would set it, isn't included.

What the excerpts do say:

- **Legacy systems:** The Regulation applies to operators of high-risk AI systems placed on the market or put into service before 2 August 2026 only if, from that date, those systems are subject to significant changes in their designs. [1] This refers to systems other than those covered by paragraph 1 of Article 111, which isn't excerpted. The mention of 2 August 2026 suggests that date matters for high-risk systems, but the excerpts don't say it is the general application date.
- **Public authorities:** Providers and deployers of high-risk AI systems intended to be used by public authorities must take the necessary steps to comply by 2 August 2030. [1]
- **Commission guidelines:** The Commission must provide guidelines on practical implementation of the classification rules, with examples of high-risk and not high-risk use cases, no later than 2 February 2026. [2]

For the general application date, you would need Article 113 itself.

Sources:
  [1] Art. 111(2)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_111
      "2. Without prejudice to the application of Article 5 as referred to in Article 113(3), point (a), this Regulation shall apply to operators of high-risk AI systems, other than the systems referred to in paragraph 1 of this Article, that have been placed on the market or put into service before 2 August 2026, only if, as from that date, those systems are subject to significant changes in their designs. In any case, the providers and deployers of high-risk AI systems intended to be used by public authorities shall take the necessary steps to comply with the requirements and obligations of this Regulation by 2 August 2030."
  [2] Art. 6(5)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_6
      "5. The Commission shall, after consulting the European Artificial Intelligence Board (the ‘Board’), and no later than 2 February 2026, provide guidelines specifying the practical implementation of this Article in line with Article 96 together with a comprehensive list of practical examples of use cases of AI systems that are high-risk and not high-risk."

claude-sonnet-5-5 · 2499 tokens in, 403 out · cost $0.0090
```

</details>

### 10. What is the VAT rate in Germany?

<details><summary>Full answer with sources and quotes ($0.0058)</summary>

```text
The excerpts don't answer this. They cover only administrative fines under the EU AI Act (Articles 99 and 100), and nothing in them mentions VAT rates in Germany or any other tax matter.

If you have a question about AI Act penalties or obligations, I'm happy to help with that.

Sources: none

claude-sonnet-5-5 · 2426 tokens in, 96 out · cost $0.0058
```

</details>
