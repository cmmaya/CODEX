## Last Version Changes:

- updated until and delay properties to Client and Provider Classes and added a condition to decide_on_contract (until > delay).
- Add Slot Class inside provider, and modified the main While to loop for each provider's Slots once a Client appears.
- Add the contract_success function: To determine if a contract was failed or cancelled.
- Change in generate_provider: now it considers a previous list of providers.
- in the Main While, slots used are now discarded for future contracts, also prefilters a list of valid_slots which fullfill the deal conditions.
- Update when a provider arrives first, to loop on Slots of the new provider and adding the record in the dict deals_per_provider

## Description

In these notes we present a simple Agent-Based Model (ABM) to simulate Codex's economy. In particular, we will propose a simple model to simulate how Clients interact with Providers, and will then proceed to discuss how to simulate potential issues that can arise in such a simple economy. For context, *Codex is a decentralized protocol and marketplace for storage provision.

Clients post storage contracts with fixed size, price, duration, and requested collateral. Storage providers compete for contracts.* The model is probabilistic in nature, i.e., it is able to capture some parameter uncertainty, as I believe that ABMs should account for several sources of uncertainty, and be used inside a Monte Carlo simulation to better understand the effects of such uncertainty.

Requirements This notebook is rather lightweight in its dependencies, and only requires very minimal (and standard!) libraries, namely numpy and matplotlib. This was done with simplicity in mind.
