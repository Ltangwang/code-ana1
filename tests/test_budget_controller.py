import pytest
import asyncio
from core.budget_controller import BudgetController
from shared.schemas import BudgetStatus


@pytest.fixture
async def budget_controller():
    controller = BudgetController(
        total_budget=100.0,
        daily_budget=10.0
    )
    await controller.initialize()
    yield controller


@pytest.mark.asyncio
async def test_initialization(budget_controller):
    status = await budget_controller.get_status()
    assert status.total_budget == 100.0
    assert status.used_budget == 0.0
    assert status.remaining_budget == 100.0


@pytest.mark.asyncio
async def test_record_expense(budget_controller):
    initial_status = await budget_controller.get_status()
    
    await budget_controller.record_expense(
        cost=5.0,
        provider="openai",
        model="gpt-4",
        tokens_used=1000
    )
    
    updated_status = await budget_controller.get_status()
    
    assert updated_status.used_budget == initial_status.used_budget + 5.0
    assert updated_status.remaining_budget == initial_status.remaining_budget - 5.0


@pytest.mark.asyncio
async def test_can_afford(budget_controller):
    assert await budget_controller.can_afford(50.0) is True
    assert await budget_controller.can_afford(150.0) is False

    await budget_controller.record_expense(
        cost=95.0,
        provider="openai",
        model="gpt-4"
    )
    
    assert await budget_controller.can_afford(10.0) is False
    assert await budget_controller.can_afford(3.0) is True


@pytest.mark.asyncio
async def test_low_budget_detection():
    controller = BudgetController(
        total_budget=100.0
    )
    await controller.initialize()

    await controller.record_expense(
        cost=85.0,
        provider="openai",
        model="gpt-4"
    )
    
    status = await controller.get_status()
    assert status.is_low_budget(threshold=0.2) is True
    assert status.remaining_percent < 0.2


@pytest.mark.asyncio
async def test_usage_history(budget_controller):
    for i in range(3):
        await budget_controller.record_expense(
            cost=1.0 + i,
            provider="openai",
            model="gpt-4"
        )
    
    history = await budget_controller.get_usage_history(days=7)
    assert len(history) == 0


@pytest.mark.asyncio
async def test_usage_by_provider(budget_controller):
    await budget_controller.record_expense(5.0, "openai", "gpt-4")
    await budget_controller.record_expense(3.0, "anthropic", "claude")
    await budget_controller.record_expense(2.0, "openai", "gpt-3.5")
    
    by_provider = await budget_controller.get_usage_by_provider()
    assert isinstance(by_provider, dict)


@pytest.mark.asyncio
async def test_adjusted_threshold(budget_controller):
    threshold = budget_controller.get_adjusted_threshold(0.6, 0.3)
    assert threshold == 0.6

    await budget_controller.record_expense(85.0, "openai", "gpt-4")
    threshold = budget_controller.get_adjusted_threshold(0.6, 0.3)
    assert threshold == 0.3

