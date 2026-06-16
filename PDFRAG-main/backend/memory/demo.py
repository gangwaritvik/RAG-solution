"""
Demo of the Group-Based Conversational Memory Architecture.

This shows how to:
1. Create conversation groups
2. Add turns to groups
3. Update summaries
4. Search groups by embedding
5. Persist to Chroma DB
"""

from backend.memory import ConversationMemoryManager, ConversationTurn


def demo_basic_operations():
    """Demo basic group memory operations."""
    
    # Initialize the manager (uses Chroma DB at CHROMA_DIR)
    manager = ConversationMemoryManager(chroma_db_path="./storage/chroma_db")
    
    print("\n" + "="*70)
    print("DEMO: Group-Based Conversational Memory Architecture")
    print("="*70)
    
    # 1. Create conversation groups for different topics
    print("\n1️⃣  Creating conversation groups...")
    group_eligibility = manager.create_conversation_group("Eligibility Requirements")
    group_emd = manager.create_conversation_group("EMD (Earnest Money Deposit)")
    group_timeline = manager.create_conversation_group("Timeline & Deadlines")
    
    print(f"   ✅ Created group: {group_eligibility.topic} (ID: {group_eligibility.group_id})")
    print(f"   ✅ Created group: {group_emd.topic} (ID: {group_emd.group_id})")
    print(f"   ✅ Created group: {group_timeline.topic} (ID: {group_timeline.group_id})")
    
    # 2. Add conversation turns to groups
    print("\n2️⃣  Adding conversation turns...")
    
    # Eligibility group turns
    manager.add_conversation_turn(
        group_eligibility.group_id,
        query="What is the minimum turnover requirement?",
        memory_summary="Minimum annual turnover ₹50 Cr required"
    )
    manager.add_conversation_turn(
        group_eligibility.group_id,
        query="Any exemptions for startups?",
        memory_summary="Startups exempted if registered <5 years; MSME relaxation at ₹25 Cr"
    )
    manager.add_conversation_turn(
        group_eligibility.group_id,
        query="What about OEM experience?",
        memory_summary="OEM experience required for 3 years with Category A supplier"
    )
    
    # EMD group turns
    manager.add_conversation_turn(
        group_emd.group_id,
        query="What is the EMD amount?",
        memory_summary="EMD = 2% of contract value, non-refundable if tender cancelled by bidder"
    )
    manager.add_conversation_turn(
        group_emd.group_id,
        query="When is EMD refunded?",
        memory_summary="EMD refunded within 30 days of contract award; retained as security deposit"
    )
    
    # Timeline group turns
    manager.add_conversation_turn(
        group_timeline.group_id,
        query="What is the bid submission deadline?",
        memory_summary="Bid submission deadline: 15th Dec 2024, 2:00 PM IST"
    )
    manager.add_conversation_turn(
        group_timeline.group_id,
        query="When are bids opened?",
        memory_summary="Technical bids opened on 16th Dec 2024, 3:00 PM IST; financial bids later"
    )
    
    print(f"   ✅ Added 3 turns to Eligibility group")
    print(f"   ✅ Added 2 turns to EMD group")
    print(f"   ✅ Added 2 turns to Timeline group")
    
    # 3. Check summarization threshold
    print("\n3️⃣  Checking group summarization status...")
    print(f"   Eligibility group unsummarized turns: {group_eligibility.unsummarized_turn_count()}")
    print(f"   Should summarize? {manager.should_summarize_group(group_eligibility.group_id)}")
    
    # 4. Update group summary
    print("\n4️⃣  Updating group summary...")
    summary = """
    Eligibility Summary:
    - Minimum turnover: ₹50 Cr annually
    - Startups exempted (<5 years)
    - MSME relaxation: ₹25 Cr
    - OEM category A experience: 3 years required
    """
    
    # In real usage, this embedding would come from Embedder
    mock_embedding = [0.1] * 1536  # Azure OpenAI embedding dimension
    manager.update_group_summary(
        group_eligibility.group_id,
        summary.strip(),
        summary_embedding=mock_embedding
    )
    
    print(f"   ✅ Updated summary for Eligibility group")
    
    # 5. List all groups
    print("\n5️⃣  Listing all conversation groups...")
    all_groups = manager.list_conversation_groups()
    for i, group in enumerate(all_groups, 1):
        print(f"   {i}. {group.topic}")
        print(f"      - Turns (recent): {len(group.recent_turns)}")
        print(f"      - Turns (all): {len(group.all_turns)}")
        print(f"      - Summary ready: {group.summary_ready}")
    
    # 6. Get group context
    print("\n6️⃣  Getting group context for answer generation...")
    context = manager.get_group_context(group_eligibility.group_id)
    if context:
        print(f"   ✅ Group: {context['topic']}")
        print(f"      Recent turns: {len(context['recent_turns'])}")
        print(f"      Summary exists: {bool(context['summary'])}")
    
    # 7. Demonstrate persistence
    print("\n7️⃣  Demonstrating persistence...")
    print(f"   Total groups in storage: {len(manager.list_conversation_groups())}")
    
    # 8. Get active group
    print("\n8️⃣  Setting and getting active group...")
    manager.set_active_group(group_eligibility.group_id)
    active = manager.get_active_group()
    if active:
        print(f"   ✅ Active group: {active.topic} (ID: {active.group_id})")
    
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70 + "\n")
    
    return manager


def demo_group_search():
    """Demo searching groups by embedding similarity."""
    
    manager = ConversationMemoryManager(chroma_db_path="./storage/chroma_db")
    
    print("\n" + "="*70)
    print("DEMO: Group Search by Embedding Similarity")
    print("="*70)
    
    groups = manager.list_conversation_groups()
    if len(groups) < 2:
        print("⚠️  Not enough groups with embeddings for search demo")
        return
    
    # Use the same mock embedding for search
    query_embedding = [0.1] * 1536
    
    print("\nSearching for groups similar to eligibility query...")
    results = manager.search_groups_by_embedding(
        query_embedding=query_embedding,
        similarity_threshold=0.3,
        top_k=3
    )
    
    if results:
        print(f"✅ Found {len(results)} relevant groups:\n")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['topic']}")
            print(f"     Similarity: {result['similarity']:.4f}")
    else:
        print("⚠️  No groups found with sufficient similarity")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run basic operations demo
    manager = demo_basic_operations()
    
    # Run search demo
    demo_group_search()
    
    print("\n💡 Next steps:")
    print("   1. Integrate query classification into the RAG pipeline")
    print("   2. Implement context resolution LLM")
    print("   3. Implement standalone query generation")
    print("   4. Create API endpoints for conversation groups")
    print("   5. Add real embeddings from Azure OpenAI Embedder")
