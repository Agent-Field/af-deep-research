"""
Tests for the deep-research-agent Agent.
Generated on 2025-07-09 16:20:06 EDT
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentfield.execution_context import ExecutionContext
from agent.agent import deepresearchagentAgent, ExampleReasonerInput, ExampleSkillInput

@pytest.fixture
def mock_agent():
    """Fixture to provide a mock instance of deepresearchagentAgent."""
    agent = deepresearchagentAgent()
    agent.context = MagicMock(spec=ExecutionContext)
    agent.context.get_config.return_value = "mock_api_key"
    return agent

@pytest.mark.asyncio
async def test_example_reasoner_success(mock_agent):
    """Test example_reasoner with a successful message processing."""
    input_data = ExampleReasonerInput(message="hello world")
    
    # Mock context.call_skill if it's used within the reasoner
    mock_agent.context.call_skill = AsyncMock(return_value={"status": "skill_done"})

    result = await mock_agent.example_reasoner(mock_agent.context, input_data)

    assert result.processed_message == "PROCESSED: HELLO WORLD"
    assert result.status == "success"
    mock_agent.context.get_config.assert_called_with("my_service.api_key", "default_api_key")

@pytest.mark.asyncio
async def test_example_skill_success(mock_agent):
    """Test example_skill with successful data processing."""
    input_data = ExampleSkillInput(data={"key": "test_value"})
    
    result = await mock_agent.example_skill(mock_agent.context, input_data)

    assert result.result == "Skill processed data: test_value"
    assert result.success is True

@pytest.mark.asyncio
async def test_example_skill_no_key(mock_agent):
    """Test example_skill when 'key' is not present in input data."""
    input_data = ExampleSkillInput(data={"another_key": "some_value"})
    
    result = await mock_agent.example_skill(mock_agent.context, input_data)

    assert result.result == "Skill processed data: no_key"
    assert result.success is True

# You can add more tests here for edge cases, error handling, etc.
