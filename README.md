## Last Version Changes:

- updated until and delay properties to Client and Provider Classes and added a condition to decide_on_contract (until > delay).
- Add Slot Class inside provider, and modified the main While to loop for each provider's Slots once a Client appears.
- Add the contract_success function: To determine if a contract was failed or cancelled.
- Change in generate_provider: now it considers a previous list of providers.
- in the Main While, slots used are now discarded for future contracts, also prefilters a list of valid_slots which fullfill the deal conditions.
- Update when a provider arrives first, to loop on Slots of the new provider and adding the record in the dict deals_per_provider
