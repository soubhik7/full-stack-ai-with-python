using System.Text.Json;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Extensions.Mcp;
using Microsoft.Extensions.Logging;

namespace Company.Function;

public class ListCustomerOrdersTool
{
    private readonly ILogger<ListCustomerOrdersTool> _logger;

    public ListCustomerOrdersTool(ILogger<ListCustomerOrdersTool> logger)
    {
        _logger = logger;
    }

    [Function(nameof(ListCustomerOrdersTool))]
    public string Run(
        [McpToolTrigger(nameof(ListCustomerOrdersTool), "List all orders for a CloudXeus customer.")]
        ToolInvocationContext context,

        [McpToolProperty("customer_id", "The customer ID")]
        string? customerId
    )
    {
        _logger.LogInformation("Tool: {Tool}", context.Name);
        _logger.LogInformation("Listing orders for customer: {CustomerId}", customerId);

        customerId ??= "UNKNOWN";

        var orders = new[]
        {
            new
            {
                order_id = "ORD-001",
                status = "Shipped"
            },
            new
            {
                order_id = "ORD-002",
                status = "Processing"
            }
        };

        return JsonSerializer.Serialize(orders);
    }
}