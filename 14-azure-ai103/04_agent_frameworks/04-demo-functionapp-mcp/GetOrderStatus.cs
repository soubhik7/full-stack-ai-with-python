using System.Text.Json;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Extensions.Mcp;
using Microsoft.Extensions.Logging;

namespace Company.Function;

public class GetOrderStatusTool
{
    private readonly ILogger<GetOrderStatusTool> _logger;

    public GetOrderStatusTool(ILogger<GetOrderStatusTool> logger)
    {
        _logger = logger;
    }

    [Function(nameof(GetOrderStatusTool))]
    public string Run(
        [McpToolTrigger(nameof(GetOrderStatusTool), "Get the current status of a CloudXeus order.")]
        ToolInvocationContext context,

        [McpToolProperty("order_id", "The order ID")]
        string? orderId
    )
    {
        _logger.LogInformation("Tool: {Tool}", context.Name);
        _logger.LogInformation("Order lookup: {OrderId}", orderId);

        orderId ??= "UNKNOWN";

        return $"Order {orderId}: Shipped. Expected delivery: 2 days.";
    }
}