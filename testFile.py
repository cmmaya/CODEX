from typing import List
import matplotlib.pyplot as plt
import numpy as np


class Client:
    """
    Creates the cient class

    """

    def __init__(
        self,
        contract_id: int,
        size: int,
        price: float,
        duration: int,
        collateral: float,
        delay: int,
    ):
        self.contract_id = contract_id
        self.size = size
        self.price = price
        self.duration = duration
        self.collateral = collateral
        self.delay = delay


def generate_clients(
    demand_factor: float,
    base_clients: int,
    size_range: tuple,
    price_range: tuple,
    duration_range: tuple,
    collateral_range: tuple,
    delay_range: tuple,
) -> List[Client]:
    '''
    Parameters:

    - demand_factor: A multiplier for adjusting the market demand.
    - base_clients: The base number of clients to generate before scaling by demand.
    - size_range: The range of storage sizes clients are seeking (min, max in GB).
    - price_range: The range of prices clients are willing to pay (min, max in CODX tokens).
    - duration_range: The range of contract durations clients are interested in (min, max in days).
    - collateral_range: The range of collateral amounts clients can provide (min, max in CODX tokens).

    Returns:
    - A list of Client objects with randomized attributes based on the input ranges.
    """

    '''
    num_clients = int(base_clients * demand_factor)
    clients = []
    for i in range(num_clients):
        contract_id = i
        size = np.random.randint(*size_range)
        price = np.random.uniform(*price_range)
        duration = np.random.randint(*duration_range)
        collateral = np.random.uniform(*collateral_range)
        delay = np.random.randint(*delay_range)
        clients.append(Client(contract_id, size, price, duration, collateral, delay))
    return clients


class Slot:
    def __init__(
        self,
        provider_id: int,
        slot_id: int,
        capacity: int,
        price_threshold: float,
        max_duration: int,
        collateral_available: float,
        until: int,
    ):
        self.provider_id = provider_id
        self.slot_id = slot_id
        self.capacity = capacity
        self.price_threshold = price_threshold
        self.max_duration = max_duration
        self.collateral_available = collateral_available
        self.until = until

    def decide_on_contract(self, contract: Client) -> bool:
        """Decides whether or not to take a contract, it should pay more than the minimum they're willing to receive and they shjould have enough collateral"""

        decision = (
            contract.price >= self.price_threshold
            and contract.duration <= self.max_duration
            and self.collateral_available >= contract.collateral
            and self.until >= contract.delay
        )

        if decision == True:
            self.collateral_available = self.collateral_available - contract.collateral

        return decision


class Provider:
    def __init__(self, provider_id: int, slots: List[Slot]):
        self.provider_id = provider_id
        self.slots = slots


def generate_providers(
    existing_providers: List[Provider],
    num_providers: int,
    capacity_range: tuple,
    price_threshold: tuple,
    duration_range: tuple,
    collateral_range: tuple,
    slots_range: tuple,
    until_range: tuple,
) -> List[Provider]:
    """
    Parameters:
    - num_providers: The number of storage providers to generate.
    - capacity_range: The range of storage capacities offered by providers (min, max in GB).
    - price_threshold: The range of minimum prices providers are willing to accept (min, max in CODX tokens).
    - duration_range: The range of maximum contract durations providers are willing to fulfill (min, max in days).
    - collateral_range: The range of collateral capacities providers can offer (min, max in CODX tokens).
    - slots_range: The range of slots a SP can own.
    - until: The range of the end date or the expiration date of the available storage offering.

    Returns:
    - A list of Provider objects with randomized attributes based on the input ranges.
    """
    for i in range(num_providers):
        slots = []
        provider_id = len(existing_providers)
        num_slots = np.random.randint(*slots_range)

        for i in range(num_slots):
            slot_id = i
            capacity = np.random.randint(*capacity_range)
            min_price = np.random.uniform(*price_threshold)
            max_duration = np.random.randint(*duration_range)
            collateral_capacity = np.random.uniform(*collateral_range)
            until = np.random.randint(*until_range)
            slots.append(
                Slot(
                    provider_id,
                    slot_id,
                    capacity,
                    min_price,
                    max_duration,
                    collateral_capacity,
                    until,
                )
            )

        existing_providers.append(Provider(provider_id, slots))
    return existing_providers


class MarketplaceBookkeeper:
    """
    This is a class to keep track of the deals made

    """

    def __init__(self):
        self.deals = []

    def add_deal(
        self,
        client: Client,
        provider: Provider,
        slot: Slot,
        size: int,
        price: float,
        duration: int,
        collateral: float,
    ):
        # Generate a unique deal ID based on the current number of deals
        deal_id = len(self.deals) + 1
        deal = Deal(deal_id, client, provider, slot, size, price, duration, collateral)
        self.deals.append(deal)
        return deal

    def get_deal_info(self, deal_id: int):
        deal = next((deal for deal in self.deals if deal.deal_id == deal_id), None)
        if deal:
            return (
                f"Deal ID: {deal.deal_id}, Client ID: {deal.client.contract_id}, "
                f"Provider ID: {deal.provider.provider_id}, Size: {deal.size}, "
                f"Price: {deal.price}, Duration: {deal.duration}, Collateral: {deal.collateral}"
            )
        return "Deal not found."


class Deal:
    def __init__(
        self,
        deal_id: int,
        client: Client,
        provider: Provider,
        slot: Slot,
        size: int,
        price: float,
        duration: int,
        collateral: float,
    ):
        """
        Initialize a new deal with details about the transaction.

        Parameters:
        - deal_id (int): A unique identifier for the deal.
        - client (Client): The client object involved in the deal.
        - provider (Provider): The provider object agreeing to fulfill the deal.
        - size (int): The size of the storage contract (e.g., in GB or TB).
        - price (float): The price agreed upon for the storage contract.
        - duration (int): The duration of the contract (e.g., in days).
        - collateral (float): The collateral amount required for the deal.
        """
        self.deal_id = deal_id
        self.client = client
        self.provider = provider
        self.slot = slot
        self.size = size
        self.price = price
        self.duration = duration
        self.collateral = collateral

    def __repr__(self):
        """
        Provide a readable representation of the Deal object for debugging and logging.
        """
        return (
            f"Deal(ID: {self.deal_id}, Client ID: {self.client.contract_id}, Provider ID: {self.provider.provider_id}, Slot ID: {self.slot.slot_id}"
            f"Size: {self.size}GB, Price: {self.price} CODX, Duration: {self.duration} days, "
            f"Collateral: {self.collateral} CODX)"
        )


# ___________________________________________________________________________________
def get_contract_success():
    """Returns False with a probability of 0.1%. Otherwise returns True"""
    rand_value = np.random.rand()
    # Return False if rand_value is less than 0.01, otherwise return True
    return rand_value >= 0.01


# Initialize simulation parameters
demand_factor = 1.0  # Initial demand factor for the marketplace
num_providers = 100  # Number of storage providers in the marketplace

# Provider capacity and contract terms ranges
capacity_range_providers = (100, 1000)  # Provider capacity range in GB
duration_range_providers = (30, 365)  # Contract duration range in days
collateral_range_providers = (0.1, 10.0)  # Collateral range in CODX tokens
price_threshold_providers = (0.5, 2)  # Minimum acceptable price for providers, in CODX tokens
slots_range = (1, 10)  # Number of slots
until_range = (4, 5)
existing_providers = []
# Client parameters
capacity_range_clients = (100, 1000)  # client capacity range in GB
duration_range_clients = (30, 365)  # Contract duration range in days
collateral_range_clients = (0.1, 5)  # Collateral range in CODX tokens
price_threshold_clients = (0.5, 3)  # Minimum acceptable price for clients, in CODX tokens

# Client arrival parameters
lambda_rate_client = 5  # Average number of clients arriving per day
lambda_rate_provider = 1 / 3  # Average number of providers arriving per day
total_time = 30  # Total simulation time in days
delay_range = (1, 3)

# Generate initial set of providers
providers = generate_providers(
    existing_providers,
    num_providers,
    capacity_range_providers,
    price_threshold_providers,
    duration_range_providers,
    collateral_range_providers,
    slots_range,
    until_range,
)

# Initialize the marketplace bookkeeper for tracking deals
bookkeeper = MarketplaceBookkeeper()

# Initialize variables for tracking simulation outcomes
unserviced_deals = 0
deals_over_time = []
collateral_over_time = []
deals_per_provider = {provider.provider_id: 0 for provider in providers}  # Deal count per provider

available_space_over_time = []
unserviced_deals_list = []
valid_slots = []
used_slots = []
# Calculate initial total capacity
total_capacity = sum(slot.capacity for provider in providers for slot in provider.slots)
available_space_over_time.append(total_capacity)

# initializes simulation
time = 0

# Main simulation loop
while time < total_time:

    # Uses a poisson process here to decide.
    t_client = np.random.exponential(1 / lambda_rate_client)
    t_provider = np.random.exponential(1 / lambda_rate_provider)

    if t_client < t_provider:  # client arrives

        time = time + t_client
        # Generate a single client based on the demand factor
        client = generate_clients(
            demand_factor,
            1,
            capacity_range_clients,
            price_threshold_clients,
            duration_range_clients,
            collateral_range_clients,
            delay_range,
        )[0]

        # Select slot with the highest collateral available
        valid_slots = [slot for provider in providers for slot in provider.slots if slot.decide_on_contract(client) if slot not in used_slots]

        # Check if there are any valid slots before calling max()
        if valid_slots:
            max_collateral_slot = max(valid_slots, key=lambda slot: slot.collateral_available)

            # Generate probability of contract being failed
            contract_success = get_contract_success()

            # Attempt to create a deal between the client and selected provider
            if max_collateral_slot and contract_success:

                # Find the provider that contains the max_collateral_slot
                max_collateral_provider = next(
                    provider for provider in providers if max_collateral_slot in provider.slots
                )

                # Add deal to the book keeper
                bookkeeper.add_deal(
                    client,
                    max_collateral_provider,
                    max_collateral_slot,
                    client.size,
                    client.price,
                    client.duration,
                    client.collateral,
                )
                used_slots.append(max_collateral_slot)

                # Update the count of deals serviced by each provider
                if max_collateral_provider.provider_id in deals_per_provider:
                    deals_per_provider[max_collateral_provider.provider_id] += 1
                else:
                    deals_per_provider[max_collateral_provider.provider_id] = 1

        else:
            unserviced_deals += 1  # Increment unserviced deals counter if no match is found
            unserviced_deals_list.append(client)

        # Record the total number of deals and collateral at this point in time
        deals_over_time.append(len(bookkeeper.deals))
        total_collateral = sum(deal.collateral for deal in bookkeeper.deals)
        collateral_over_time.append(total_collateral)

    else:  # provider arrives
        time = time + t_provider
        new_provider = generate_providers(
            providers[:],
            1,
            capacity_range_providers,
            price_threshold_providers,
            duration_range_providers,
            collateral_range_providers,
            slots_range,
            until_range,
        )[-1]
        providers.append(new_provider)

        total_capacity += sum(slot.capacity for slot in new_provider.slots)  # Increase total capacity

        # Attempt to service previously unserviced deals
        for unserviced_client in unserviced_deals_list[:]:  # Iterate over a copy of the list
            for slot in new_provider.slots:
                if slot.decide_on_contract(unserviced_client):
                    bookkeeper.add_deal(
                        unserviced_client,
                        new_provider,
                        slot,
                        unserviced_client.size,
                        unserviced_client.price,
                        unserviced_client.duration,
                        unserviced_client.collateral,
                    )
                    unserviced_deals_list.remove(unserviced_client)  # Remove serviced client from the unserviced list                    

                    # Check if the provider_id exists in the dictionary and update accordingly
                    if new_provider.provider_id in deals_per_provider:
                        deals_per_provider[new_provider.provider_id] += 1
                    else:
                        deals_per_provider[new_provider.provider_id] = 1

                    break  # Exit the inner loop as the client has been serviced
            else:
                continue  # If the inner loop wasn't broken, continue with the next client
            break

    current_collateral = sum(deal.collateral for deal in bookkeeper.deals if deal.provider in providers)
    available_space_over_time.append(total_capacity - current_collateral)

# Output the total number of deals serviced and unserviced deals
print(f"Total deals serviced: {len(bookkeeper.deals)}")
print(f"Unserviced deals: {unserviced_deals}")

# Plot the total number of deals and collateral over time
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.plot(deals_over_time, label="Total Deals")
plt.xlabel("Day")
plt.ylabel("Total Deals")
plt.title("Total Deals Over Time")
plt.legend()

plt.subplot(1, 3, 2)

plt.plot(collateral_over_time, label="Total Collateral", color="orange")
plt.xlabel("Day")
plt.ylabel("Total Collateral")
plt.title("Total Collateral Over Time")
plt.legend()

plt.subplot(1, 3, 3)

plt.plot(available_space_over_time, label="Available Space", color="green")
plt.xlabel("Time")
plt.ylabel("Available Space (GB)")
plt.title("Historical Available Space Over Time")
plt.legend()
plt.tight_layout()

plt.show()

# Plot the distribution of deals per provider
plt.figure(figsize=(16, 9))
providers_ids = list(deals_per_provider.keys())
deals_counts = list(deals_per_provider.values())

plt.bar(providers_ids, deals_counts, color='skyblue')
plt.xlabel('Provider ID')
plt.ylabel('Number of Deals Serviced')
plt.title('Distribution of Deals per Provider')
plt.xticks(providers_ids, rotation=45)  # Ensure provider IDs are readable
plt.show()