import os
import re
from enum import Enum

from datasets import Dataset, DatasetDict

from .. import PlicitType, SpanTags, SpanTagsFormat


def split_train_valid_test_dataset(
    ds: Dataset, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    # train: 80%, valid: 10%, test: 10%
    dsd_train_valtest: DatasetDict = ds.train_test_split(
        test_size=0.2, shuffle=True, seed=seed
    )
    ds_train = dsd_train_valtest["train"]
    dsd_val_test: DatasetDict = dsd_train_valtest["test"].train_test_split(
        test_size=0.5, shuffle=True, seed=seed
    )
    ds_valid = dsd_val_test["train"]
    ds_test = dsd_val_test["test"]
    return ds_train, ds_valid, ds_test


def is_explicit(text: str, cause: str, effect: str) -> bool:
    lst_pattern: list[str] = [
        "{cause} where {effect}",
        "Where {cause}, {effect}" "{effect} amid {cause}",
        "{effect} amidst {cause}",
        "amid {cause}, {effect}" "amist {cause}, {effect}",
        "With {cause}, {effect}",
        "{effect}, with {cause}",
        "Without {cause}, {effect}",
        "Having {cause}, {effect}",
        "{effect} every time {cause}",
        "every time {cause}, {effect}",
        "whenever {cause}, {effect}",
        "{cause} whenever {effect}",
        "Anytime {cause}, {effect}",
        "Any time {cause}, {effect}",
        "{effect}, as {cause}",
        "As {cause}, {effect}",
        "If {cause}, {effect}",
        "{effect} if {cause}",
        "if {cause}, then {effect}",
        "should {cause}, {effect}",
        "{effect} should {cause}",
        "{cause} to {effect}",
        "{cause} for {effect}",
        "Once {cause}, {effect}",
        "{effect} once {cause}",
        "{effect} in advance of {cause}",
        "in advance of {cause}, {effect}",
        "in the aftermath of {cause}, {effect}",
        "{effect} in the aftermath of {cause}",
        "{effect} in the wake of {cause}",
        "in the wake of {cause}, {effect}",
        "{effect}, since {cause}",
        "Since {cause}, {effect}",
        "{cause}, and then {effect}",
        "Then, {effect}",
        "the aftermath of {cause} is {effect}",
        "{effect} at {cause}",
        "At {cause}, {effect}",
        "Before {effect}, {cause}",
        "{cause} before {effect}",
        "following {cause}, {effect}",
        "upon {cause}, {effect}",
        "{effect} upon {cause}",
        "on {cause}, {effect}",
        "{effect} on {cause}",
        "after {cause}, {effect}",
        "{effect} after {cause}",
        "Before {cause}, {effect}",
        "{effect} before {cause}",
        "{effect} until {cause}",
        "during {cause}, {effect}",
        "while {cause}, {effect}",
        "{effect} will take {cause}" "when {cause}, {effect}",
        "{cause} when {effect}",
        "{effect} as long as {cause}",
        "{cause}, and consequently, {effect}",
        "Consequently, {effect}",
        "{cause}; hence, {effect}",
        "Hence {effect}.",
        "{cause}; therefore, {effect}",
        "Therefore, {effect}.",
        "For {effect} to {effect}, {cause}",
        "for {effect}, {cause}",
        "in order to {effect}, {cause}",
        "As a result, {effect}",
        "{cause}, and as a result, {effect}",
        "{effect} as a result of {cause}",
        "{effect} by virtue of {cause}",
        "{effect} for the purpose of {cause}",
        "for the purpose of {cause}, {effect}",
        "{effect} for purposes of {cause}",
        "{effect} for {cause}",
        "{effect} for the reason that {cause}",
        "{effect} for reasons of {cause}",
        "{effect} in case {cause}",
        "{effect} in case of {cause}",
        "{effect} in an effort to {cause}",
        "In an effort to {cause}, {effect}",
        "{effect} in response to {cause}",
        "In response, {effect}",
        "In the face of {cause}, {effect}",
        "in view of {cause}, {effect}",
        "{effect} on grounds of {cause}",
        "{effect} on the grounds of {cause}",
        "{effect} on the grounds that {cause}",
        "{effect} on {cause}",
        "{effect} so as to {cause}",
        "{effect} on condition of {cause}",
        "{cause}, so {effect}",
        "{cause}, and thus {effect}",
        "Thus, {effect}",
        "{effect} to {cause}",
        "To {cause}, {effect}",
        "In order to {cause}, {effect}",
        "{effect} in order to {cause}",
        "{effect} because {cause}",
        "because {cause}, {effect}",
        "{effect}, because {cause}",
        "{effect}, for {cause}",
        "Given {cause}, {effect}",
        "Given that {cause}, {effect}",
        "{effect}, given that {cause}",
        "In an attempt to {cause}, {effect}",
        "{effect}, in an attempt to {cause}",
        "{effect} lest {cause}",
        "Now that {cause}, {effect}",
        "{effect} so {cause}",
        "So that {cause}, {effect}",
        "So {cause}, {effect}",
        "{effect} so that {cause}",
        "{effect} thanks to {cause}",
        "Thanks to {cause}, {effect}",
        "{cause} else {effect}",
        "{effect} unless {cause}",
        "Unless {cause}, {effect}",
        "Unless {cause}, then {effect}",
        "the implications of {cause} are {effect}",
        "{effect} because of {cause}",
        "because of {cause}, {effect}",
        "{effect} by {cause}",
        "By {cause}, {effect}",
        "{effect} by reason of {cause}",
        "For {cause}, {effect}",
        "{effect} for the sake of {cause}",
        "For the sake of {cause}, {effect}",
        "{effect} from {cause}",
        "{effect} from {cause}",
        "Given {cause}, {effect}",
        "{effect} given {cause}",
        "{effect} in {cause}",
        "In light of {cause}, {effect}",
        "{cause} in light of {effect}",
        "{cause} into {effect}",
        "{effect} of {cause}",
        "{effect} out of {cause}",
        "{cause} to {effect}",
        "{effect} with {cause}",
        "{effect}, barring {cause}",
        "barring {cause}, {effect}",
        "reason to {effect}",
        "{cause} has {effect}",
        "{cause} had {effect}",
        "{cause} has had {effect}",
        "{cause} had had {effect}",
        "{cause}, giving {effect}",
    ]
    lst_ptn_verb: list[str] = [
        "{effect} arises from {cause}",
        "{effect} arose from {cause}",
        "{effect} (has |had )?arisen from {cause}",
        "{cause} brings on {effect}",
        "{cause} (has |had )?brought on {effect}",
        "{cause} creates {effect}",
        "{cause} (has |had )?created {effect}",
        "{cause} produces {effect}",
        "{cause} (has |had )?produced {effect}",
        "{cause} engenders {effect}",
        "{cause} (has |had )?engendered {effect}",
        "{cause} generates {effect}",
        "{cause} (has |had )?generated {effect}",
        "{cause} gives rise to {effect}",
        "{cause} gave rise to {effect}",
        "{cause} (has|had) given rise to {effect}",
        "{cause} incites {effect}",
        "{cause} (has |had )?incited {effect}",
        "{cause} launches {effect}",
        "{cause} (has |had )?launched {effect}",
        "{cause} sets off {effect}",
        "{cause} (has |had )?set off {effect}",
        "{effect} stems from {cause}",
        "{effect} (has |had )?stemmed from {cause}",
        "{cause} triggers {effect}",
        "{cause} (has |had )?triggered {effect}",
        "{cause} sparks {effect}",
        "{cause} (has |had )?sparked {effect}",
        "{cause} precipitates {effect}",
        "{cause} (has |had )?precipitated {effect}",
        "{cause} eliminates {effect}",
        "{cause} (has |had )?eliminated {effect}",
        "{cause} allows {effect}",
        "{cause} (has |had )?allowed {effect}",
        "{cause} compels {effect}",
        "{cause} (has |had )?compelled {effect}",
        "{cause} forces {effect}",
        "{cause} (has |had )?forced {effect}",
        "{cause} permits {effect}",
        "{cause} (has |had )?permitted {effect}",
        "{cause} requires {effect}",
        "{cause} (has |had )?required {effect}",
        "{cause} forbids {effect}",
        "{cause} forbade {effect}",
        "{cause} (has|had) forbidden {effect}",
        "{cause} prevents {effect}",
        "{cause} (has |had )?prevented {effect}",
        "{cause} prohibits {effect}",
        "{cause} (has |had )?prohibited {effect}",
        "{effect} comes after {cause}",
        "{effect} (has |had )?came after {cause}",
        "{effect} follows {cause}",
        "{effect} (has |had )?followed {cause}",
        "{cause} clears the way for {effect}",
        "{cause} (has |had )?cleared the way for {effect}",
        "{cause} opens the way for {effect}",
        "{cause} (has |had )?opened the way for {effect}",
        "{cause} opens the way to {effect}",
        "{cause} (has |had )?opened the way to {effect}",
        "{cause} opens the door for {effect}",
        "{cause} (has |had )?opened the door for {effect}",
        "{cause} opens the door to {effect}",
        "{cause} (has |had )?opened the door to {effect}",
        "{cause} paves the way for {effect}",
        "{cause} (has |had )?paved the way for {effect}",
        "attribute(s|d)? {effect} to {cause}",
        "blame(s|d)? {cause} for {effect}",
        "blame(s|d)? {effect} on {cause}",
        "{cause} causes {effect}",
        "{cause} (has |had )?caused {effect}",
        "{effect} comes from {cause}",
        "{effect} (has |had )?came from {cause}",
        "{cause} contributes to {effect}",
        "{cause} (has |had )?contributed to {effect}",
        "{effect} depends on {cause}",
        "{effect} (has |had )?depended on {cause}",
        "{cause} drives {effect}",
        "{cause} drove {effect}",
        "{cause} (has|had) driven {effect}",
        "{cause} eases {effect}",
        "{cause} (has |had )?eased {effect}",
        "{cause} enables {effect}",
        "{cause} (has |had )?enabled {effect}",
        "{cause} encourages {effect}",
        "{cause} (has |had )?encouraged {effect}",
        "{cause} facilitates {effect}",
        "{cause} (has |had )?facilitated {effect}",
        "{cause} feeds {effect}",
        "{cause} (has |had )?fed {effect}",
        "{cause} fosters {effect}",
        "{cause} (has |had )?fostered {effect}",
        "{cause} helps {effect}",
        "{cause} (has |had )?helped {effect}",
        "{cause} helps to {effect}",
        "{cause} helped to {effect}",
        "{cause} (has |had )?helped to {effect}",
        "{cause} induces {effect}",
        "{cause} (has |had )?induced {effect}",
        "{cause} inhibits {effect}",
        "{cause} (has |had )?inhibited {effect}",
        "{cause} leads to {effect}",
        "{cause} (has |had )?led to {effect}",
        "{cause} makes for {effect}",
        "{cause} (has |had )?made for {effect}",
        "{cause} means {effect}",
        "{cause} (has |had )?meant {effect}",
        "{cause} means that {effect}",
        "{cause} (has |had )?meant that {effect}",
        "{cause} necessitates {effect}",
        "{cause} (has |had )?necessitated {effect}",
        "{cause} promotes {effect}",
        "{cause} (has |had )?promoted {effect}",
        "{cause} prompts {effect}",
        "{cause} (has |had )?prompted {effect}",
        "{cause} provokes {effect}",
        "{cause} (has |had )?provoked {effect}",
        "{effect} requires {cause}",
        "{effect} (has |had )?required {cause}",
        "{effect} results from {cause}",
        "{effect} (has |had )?resulted from {cause}",
        "{effect} results",
        "{effect} (has |had )?resulted",
        "{cause} results in {effect}",
        "{cause} (has |had )?resulted in {effect}",
        "{cause} spurs {effect}",
        "{cause} (has |had )?spurred {effect}",
        "{cause} inspires {effect}",
        "{cause} (has |had )?inspired {effect}",
        "{cause} gives {effect}",
        "{cause} gave {effect}",
        "{cause} (has|had) given {effect}",
        "{cause} averts {effect}",
        "{cause} (has |had )?averted {effect}",
        "{cause} avoids {effect}",
        "{cause} (has |had )?avoided {effect}",
        "{cause} blocks {effect}",
        "{cause} (has |had )?blocked {effect}",
        "{cause} deters {effect}",
        "{cause} (has |had )?deterred {effect}",
        "{cause} discourages {effect}",
        "{cause} (has |had )?discouraged {effect}",
        "{cause} foils {effect}",
        "{cause} (has |had )?foiled {effect}",
        "{cause} hampers {effect}",
        "{cause} (has |had )?hampered {effect}",
        "{cause} hinders {effect}",
        "{cause} (has |had )?hindered {effect}",
        "{cause} impedes {effect}",
        "{cause} (has |had )?impeded {effect}",
        "{cause} wards off {effect}",
        "{cause} (has |had )?warded off {effect}",
        "{effect} takes {cause}",
        "{effect} took {cause}",
        "{effect} (has|had) taken {cause}",
        "it takes {cause} to {effect}"
        "it took {cause} to {effect}"
        "it (has|had) taken {cause} to {effect}"
        "{cause} ensures {effect}",
        "{cause} (has |had )?ensured {effect}",
        "{cause} ensures that {effect}",
        "{cause} (has |had )?ensured that {effect}",
        "{cause} makes sure {effect}",
        "{cause} (has |had )?made sure {effect}",
        "{cause} makes sure that {effect}",
        "{cause} (has |had )?made sure that {effect}",
        "{cause} guarantees {effect}",
        "{cause} (has |had )?guaranteed {effect}",
        "{cause} guarantees that {effect}",
        "{cause} (has |had )?guaranteed that {effect}",
        "{cause} makes certain {effect}",
        "{cause} (has |had )?made certain {effect}",
        "{cause} makes certain that {effect}",
        "{cause} (has |had )?made certain that {effect}",
        "{cause} assures {effect}",
        "{cause} (has |had )?assured {effect}",
        "{cause} assures that {effect}",
        "{cause} (has |had )?assured that {effect}",
        "{cause} thwarts {effect}",
        "{cause} (has |had )?thwarted {effect}",
    ]
    lst_pattern.extend(lst_ptn_verb)

    lst_ptn_is_and_det: list[str] = [
        "{effect} is conditioned on {cause}",
        "{effect} is conditional on {cause}",
        "{effect} is contingent on {cause}",
        "{cause} is critical to {effect}",
        "{cause} is essential to {effect}",
        "{cause} is responsible for {effect}",
        "{cause} is vital to {effect}",
        "{cause} is vital for {effect}",
        "{cause} is why {effect}",
        "{cause} is grounds for {effect}",
        "{cause} is the key to {effect}",
        "{cause} is the reason why {effect}",
        "{cause} is the reason for {effect}",
        "{cause} is the condition of {effect}",
        "{cause} is the condition for {effect}",
        "{effect} is predicated on {cause}",
        "{effect}, for DET purposes?: {cause}",
        "For DET reason {cause}, {effect}",
        "{effect}, for DET reasons?: {cause}",
        "{effect} with DET goal of {cause}",
        "{effect} with DET objective of {cause}",
        "{cause} is DET cause of {effect}",
        "DET consequence of {cause} is {effect}",
        "DET consequence is {effect}",
        "DET effect of {cause} is {effect}",
        "{cause} has DET effect: {effect}",
        "DET effect is (that )?{effect}",
        "{cause} is DET necessary condition of {effect}",
        "{cause} is DET necessary condition for {effect}",
        "DET reason (that )?{effect} is {cause}",
        "DET reason (that )?{effect} is because {cause}",
        "DET reason for {effect} is {cause}",
        "DET reason why {effect} is {cause}",
        "{effect} is DET result of {cause}",
        "DET result is {effect}",
    ]
    lst_ptn_is_and_det = list(
        map(lambda x: x.replace("DET ", "(a |an |the )?"), lst_ptn_is_and_det)
    )
    lst_ptn_is_vars: list[str] = sum(
        [
            list(
                map(
                    lambda y: y.replace(" is ", f" {z} "),
                    filter(lambda x: " is " in x, lst_ptn_is_and_det),
                )
            )
            for z in ("was", "has been", "had been")
        ],
        [],
    )
    lst_ptn_has_vars: list[str] = sum(
        [
            list(
                map(
                    lambda y: y.replace(" has ", f" {z} "),
                    filter(lambda x: " has " in x, lst_ptn_is_and_det),
                )
            )
            for z in ("has had", "had", "had had")
        ],
        [],
    )
    lst_pattern.extend(lst_ptn_is_and_det)
    lst_pattern.extend(lst_ptn_is_vars)
    lst_pattern.extend(lst_ptn_has_vars)

    return any(
        [
            re.match(
                p.format(cause=re.escape(cause), effect=re.escape(effect)).lower(),
                text.lower(),
            )
            for p in lst_pattern
        ]
    )


def wrapper_is_explicit(example: dict[str, str]) -> bool:
    text: str = example["text"]
    tagged_text: str = example["tagged_text"]
    bool_explicit: bool
    if SpanTags.cause_begin in tagged_text:
        bool_explicit = is_explicit(
            text,
            cause=re.search(
                f"{SpanTags.cause_begin}(.*?){SpanTags.cause_end}", tagged_text
            ).group(1),
            effect=re.search(
                f"{SpanTags.effect_begin}(.*?){SpanTags.effect_end}", tagged_text
            ).group(1),
        )
    else:
        # multiple
        bool_explicit = True
        i: int = 1
        while SpanTagsFormat.cause_begin.format(i) in tagged_text:
            if not is_explicit(
                text,
                cause=re.search(
                    (
                        f"{SpanTagsFormat.cause_begin.format(i)}(.*?)"
                        + SpanTagsFormat.cause_end.format(i)
                    ),
                    tagged_text,
                ).group(1),
                effect=re.search(
                    (
                        f"{SpanTagsFormat.effect_begin.format(i)}(.*?)"
                        + SpanTagsFormat.effect_end.format(i)
                    ),
                    tagged_text,
                ).group(1),
            ):
                bool_explicit = False
                break
            i += 1
    return bool_explicit


def filter_plicit_dataset(ds: Dataset, plicit_enum: Enum) -> Dataset:
    num_proc: int = min(os.cpu_count() - 2, 16)
    if plicit_enum == PlicitType.explicit:
        ds = ds.filter(wrapper_is_explicit, num_proc=num_proc)
    elif plicit_enum == PlicitType.implicit:
        ds = ds.filter(
            lambda example: not wrapper_is_explicit(example), num_proc=num_proc
        )
    else:
        assert plicit_enum == PlicitType.all
    return ds
