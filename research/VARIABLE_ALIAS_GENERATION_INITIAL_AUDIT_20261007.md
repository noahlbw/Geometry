# Initial metadata-only Qwen generation inspection

The fixed pinned language source is generating240 foreground categories across VDD, Potsdam, VOC21, PC60 and ADE150. No target images, masks or model scores enter generation. Raw responses are preserved; the existing best segmentation model is unchanged.

The first responses show that schema-valid output is not sufficient semantic validation. Potsdam impervious surface includes `rooftop`; car includes `truck` and `bus`; VDD water includes `floodplain`, `marsh` and `swamp`. Some are parts, broad categories or mixed land-cover concepts, not necessarily exclusive instances of the annotated parent. The source also calls trucks and buses synonyms of vehicle, confusing synonyms with subtypes. These are metadata/sense concerns, not measured visual harm.

Decision: finish and preserve the one frozen generation, but do not promote its structurally accepted list as a semantically clean vocabulary. A typed Qwen response is a claim, not a verified ontology relation. Exact competitor-name and duplicate filtering cannot catch indirect competitor denotations. A subsequent admissibility rule needs taxonomy meaning/ownership, not only string overlap or more prompt instructions. No manual per-class correction or target-score selection is performed during this run.

The source can still serve as a reproducible candidate pool for a separately frozen admission experiment. Variable counts additionally alter the inherited wide query-count prior, so a downstream comparison must separate semantic admission from count/prior effects rather than attribute every gain to alias quality.
