## Rules of Coding

1. Docstrings: 

- All modules should have docstrings for a. what the module does, b. what entities in it, and c. how they are used by other modules.

- All entities/class should have docstrings for a. tight description of what the entity does, b. what attributes it has, and c. what methods it has.

- All functions/methods should have docstrings for a. tight description of what the function does, b. what parameters it takes and types, and c. what it returns and types.

2. Variable Naming:

- All modules, entities, functions, variables, parameters should be named a. easily understandable, b. descriptive, c. consistent with the rest of the codebase.

- All modules, entities, functions, variables, parameters are forbidden to be named with a. single letter, b. shorthands, c. abbreviations, d. placeholder names, e. hardwired names, f. names that indicates overlapping meaning with other names, g. names that are too abstract, too generic or vague, too narrow or too specific.

3. Structures and Respectting to contracts:

- When a module is proposed in the plan, they should be carried out and relative items should be storaged correctly.

You should NOT put entity B inside module A for convenience. 

Bad Example: 
`
/tests/test_module_a.py
/gui/test.py
`

In this case, test codes should be in test folder, not scattered in other folders. 

- When a module is proposed in the plan, you should implement it, rather than a. ignore it because of convenience or you think it is not important, or b. implement implicity or conflating with other modules.

You should respect the hard contracts in this plan.

Bad Example:
`
Plan: you need to have a /schema
execution: /gui/schema.py
`
In this case, the agent made a decition that it has no authority, that hiding the schema in gui folder is more convenient.

- When an existing entity or function or parameter is proposed or implemented, you should respect the interface, rather than reivent a new one, or modify the existing for one case usage.

You should reuse and respect, rather than duplicate overlapping functionality by treating each case as special case. This requires abstraction.

Bad Example:

`
class create schema_1():
class create schema_2():
class create schema_3():
`

In this case, the agent failed to abstract the functionality and instead treated each case as special case.

4. Logical errors:

- Hard forbidden to fallback logics. If there is an error, they should be correctly break and report the exact error, rather than catching, fallback, and continue.

Instead, they should be expected to be handled correctly.

- Hard forbidden on placeholders and hardwires. 

Instead, when plan is telling you to make them configurable, they should be configurable.

- Hard forbidden on decrorative code, orphanated codes, or shortcircuits. 

If the user instructs to add a functionality, they should be correctly handled upstream and downstream, rather than orphanated and never used. 

- Hard forbidden on short circuiting and hardwiring when testing. They should NOT a. call the entities and see if they exist, then claim success, or b. check if the code is executed, then claim success, c. short circuiting passing parameters to avoid real usage scenarios, d. hardiwre successful cases.

Instead, all tests must be end to end. Real values need to be passed in and tested. The input and output should represent real usage scenarios.

- Hard forbidden on workarounds, hacks, dirty code, shortcuts, and other tricks. All solutions must be elegant AND maintainable.