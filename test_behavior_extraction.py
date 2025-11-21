"""Test behavior extraction system with real episode data."""

import json
import os
import sys
from pathlib import Path

# Set up environment
os.environ.setdefault('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))

def test_behavior_extraction():
    """Test the behavior extraction system."""

    # Load actual episodes
    episodes_path = Path('runs/marl_2025-11-21_00-03-12/episodes.json')
    print(f'Loading episodes from: {episodes_path}')

    with open(episodes_path, 'r', encoding='utf-8') as f:
        episodes_data = json.load(f)

    print(f'Loaded {len(episodes_data)} episodes')

    # Sort by reward
    sorted_episodes = sorted(episodes_data, key=lambda x: x['total_reward'], reverse=True)

    # Get top 20% (4 out of 20)
    top_count = max(1, int(len(sorted_episodes) * 0.2))
    top_episodes = sorted_episodes[:top_count]

    print(f'\nTop {top_count} episodes for analysis:')
    for i, ep in enumerate(top_episodes, 1):
        print(f'  {i}. Reward: {ep["total_reward"]:.2f}, Turns: {len(ep["turns"])}')

    # Analyze turn structure
    print(f'\nAnalyzing turn structure of top episode:')
    top_ep = top_episodes[0]
    agent_roles = set()

    for turn in top_ep['turns']:
        agent_roles.add(turn['agent_role'])

    print(f'  Agent roles present: {agent_roles}')

    # Show sample of each agent's contribution
    print(f'\nSample contributions from each agent:')
    for role in ['literature_synthesizer', 'hypothesis_generator', 'experimental_designer',
                 'data_analyst', 'paper_writer']:
        for turn in top_ep['turns']:
            if turn['agent_role'] == role:
                action = turn['action']
                preview = action[:150].replace('\n', ' ')
                print(f'\n  {role}:')
                print(f'    Length: {len(action)} chars')
                print(f'    Preview: {preview}...')
                break

    # Now test actual behavior extraction
    print(f'\n{"="*60}')
    print('Testing Behavior Extraction with Gemini API')
    print(f'{"="*60}')

    # Check API key (try env first, then load from .env file)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('GEMINI_API_KEY='):
                        api_key = line.strip().split('=', 1)[1]
                        print('Loaded API key from .env file')
                        break
        except:
            pass

    if not api_key:
        print('ERROR: GEMINI_API_KEY not found in env or .env file')
        return False

    print(f'API key found: {api_key[:10]}...')

    # Import and test
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Create simple test prompt
        agent_roles = ['literature_synthesizer', 'hypothesis_generator',
                      'experimental_designer', 'data_analyst', 'paper_writer']

        # Build a minimal prompt
        prompt = f"""You are analyzing successful multi-agent research_lab episodes to extract behavioral patterns.

Here is 1 top-performing episode (Reward: {top_episodes[0]['total_reward']:.2f}):

"""
        # Add first 3 turns from top episode
        for i, turn in enumerate(top_episodes[0]['turns'][:3], 1):
            prompt += f"\nTurn {i} ({turn['agent_role']}): {turn['action'][:500]}...\n"

        prompt += """

Your task: Identify what made this episode successful. Extract 2-3 specific behavioral patterns for the literature_synthesizer role.

Return your analysis as a JSON object with this exact structure:
{
  "literature_synthesizer": {
    "collaboration": ["pattern 1", "pattern 2"],
    "scientific_rigor": ["pattern 1"],
    "novelty": ["pattern 1"]
  }
}

Return ONLY the JSON object, no other text."""

        print(f'\nSending test prompt to Gemini ({len(prompt)} chars)...')

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048
            )
        )

        print(f'\nReceived response from Gemini!')
        print(f'Response text length: {len(response.text)} chars')
        print(f'\nResponse preview:')
        print(response.text[:500])

        # Try to parse JSON
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            behaviors = json.loads(json_match.group(0))
            print(f'\nSuccessfully parsed JSON!')
            print(f'Behaviors extracted:')
            print(json.dumps(behaviors, indent=2))
            return True
        else:
            print(f'\nWARNING: Could not find JSON in response')
            return False

    except Exception as e:
        print(f'\nERROR during behavior extraction: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_behavior_extraction()
    sys.exit(0 if success else 1)
